import warnings
from pathlib import Path
from typing import Literal, Self, cast

import PIL.Image
import tensordict
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
import transformers
from loguru import logger
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput
from transformers.models.convnext.configuration_convnext import ConvNextConfig
from transformers.models.convnext.modeling_convnext import ConvNextModel

import music_ocr.tokenizer
import music_ocr.utils
from music_ocr.model.base import ArchitectureConfig as BaseArchitectureConfig
from music_ocr.model.base import Preprocessor as BasePreprocessor
from music_ocr.model.base import PreprocessorConfig as BasePreprocessorConfig
from music_ocr.model.base import build_model as _build_model
from music_ocr.model.base import build_preprocessor as _build_preprocessor


class ArchitectureConfig(BaseArchitectureConfig):
    kind: Literal["smt"] = "smt"
    # extra fields are passed to SMTConfig


class PreprocessorConfig(BasePreprocessorConfig):
    kind: Literal["smt"] = "smt"
    tokenizer: music_ocr.tokenizer.SpaceTokenizerConfig
    img_size: tuple[int, int]
    pad_to_multiple_of: int | None = 16

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        return Preprocessor.load(path)


class Preprocessor:
    class SingleInput(tensordict.TensorClass):
        pixel_values: torch.Tensor
        labels: torch.Tensor | None = None

    class Inputs(tensordict.TensorClass):
        pixel_values: torch.Tensor
        labels: torch.Tensor | None = None

    def __init__(self, config: PreprocessorConfig, tokenizer: music_ocr.tokenizer.SpaceTokenizer) -> None:
        assert isinstance(tokenizer, music_ocr.tokenizer.Tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        self.image_size = config.img_size
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.sep_token_id = tokenizer.get_token_id("<SEP>")

    @classmethod
    def from_vocab(cls, config: PreprocessorConfig, vocab: list[str]) -> Self:
        tokenizer = music_ocr.tokenizer.SpaceTokenizer(vocab, config.tokenizer)
        return cls(config, tokenizer)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(self.config.model_dump_json(indent=2), encoding="utf-8")
        self.tokenizer.save(path / "tokenizer.json")

    @classmethod
    def load(cls, path: Path) -> Self:
        path = Path(path)
        config = PreprocessorConfig.model_validate_json((path / "config.json").read_text(encoding="utf-8"))
        tokenizer = music_ocr.tokenizer.SpaceTokenizer.load(path / "tokenizer.json")
        return cls(config, tokenizer)

    def preprocess_one(self, image: PIL.Image.Image, label: str | None) -> SingleInput:
        image = image.convert("L").resize(self.image_size, resample=PIL.Image.Resampling.BICUBIC)
        pixels = torchvision.transforms.functional.to_tensor(image)
        if label is None:
            return self.SingleInput(pixel_values=pixels)
        labels_list = [self.sep_token_id] + self.tokenizer.encode(label) + [self.eos_token_id]
        labels = torch.tensor(labels_list, dtype=torch.long)
        return self.SingleInput(pixel_values=pixels, labels=labels)

    def process_batch(self, batch: list[SingleInput]) -> tensordict.TensorDict:
        images = torch.stack([item.pixel_values for item in batch], dim=0)
        collated = self.Inputs(pixel_values=images, labels=None)
        if batch[0].labels is not None:
            assert all(item.labels is not None for item in batch), "Either all or none of the examples must have labels"
            collated.labels = music_ocr.utils.pad_sequence(
                [item.labels for item in batch],  # type: ignore
                batch_first=True,
                padding_value=-100,
                padding_side="right",
                pad_to_multiple_of=self.config.pad_to_multiple_of,
            )
        outputs = collated.to_tensordict()
        return cast(tensordict.TensorDict, outputs)

    def __call__(
        self, image: list[PIL.Image.Image], label: list[str] | list[None] | None = None
    ) -> tensordict.TensorDict:
        if label is None:
            label = [None] * len(image)
        return self.process_batch([self.preprocess_one(img, lbl) for img, lbl in zip(image, label)])


class SMTConfig(transformers.PretrainedConfig):
    model_type = "SMT"
    has_no_defaults_at_init = True

    def __init__(
        self,
        maxh,
        maxw,
        maxlen,
        vocab_size,
        in_channels,
        d_model,
        dim_ff,
        num_hidden_layers,
        attn_heads,
        use_flash_attn,
        decoder_start_token_id,
        pad_token_id,
        eos_token_id,
        **kwargs,
    ):
        super().__init__(is_encoder_decoder=True, **kwargs)
        self.architectures = ["SMT"]
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.maxh = maxh
        self.maxw = maxw
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.dim_ff = dim_ff
        self.num_attn_heads = attn_heads
        self.num_hidden_layers = num_hidden_layers
        self.use_flash_attn = use_flash_attn


class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, dim, h_max, w_max):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim

        self.pe = torch.nn.Buffer(torch.zeros((dim, h_max, w_max), requires_grad=False), persistent=False)
        div = torch.exp(-torch.arange(0.0, dim // 2, 2) / dim * torch.log(torch.tensor(1e4))).unsqueeze(1)
        w_pos = torch.arange(0.0, w_max) * div
        h_pos = torch.arange(0.0, h_max) * div
        self.pe[: dim // 2 : 2] = torch.sin(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[1 : dim // 2 : 2] = torch.cos(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[dim // 2 :: 2] = torch.sin(w_pos).unsqueeze(1).repeat(1, h_max, 1)
        self.pe[dim // 2 + 1 :: 2] = torch.cos(w_pos).unsqueeze(1).repeat(1, h_max, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add 2D positional encoding to x
        x: Tensor(B, C, H, W)
        returns:
        - Tensor(B, C, H, W)
        """
        return x + self.get_pe_by_size(x.size(-2), x.size(-1))

    def get_pe_by_size(self, h: int, w: int) -> Tensor:
        return self.pe[:, :h, :w]


class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, dim, len_max):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.nn.Buffer(torch.zeros((len_max, dim), requires_grad=False), persistent=False)
        div = torch.exp(-torch.arange(0.0, dim, 2) / dim * torch.log(torch.tensor(1e4)))
        l_pos = torch.arange(0.0, len_max).unsqueeze(1) * div
        self.pe[:, ::2] = torch.sin(l_pos)
        self.pe[:, 1::2] = torch.cos(l_pos)

    def forward(self, x: Tensor, start: int = 0) -> Tensor:
        """
        Add 1D positional encoding to x
        x: Tensor(B, L, C)
        start: index for x[:, 0, :]
        returns:
        - Tensor(B, L, C)
        """
        if isinstance(start, int):
            return x + self.pe[start : start + x.size(-2)]
        else:
            result = x.clone()
            for i in range(x.size(0)):
                result[i] = x[i] + self.pe[start[i] : start[i] + x.size(-2)]
            return result


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()

        assert d_model % num_heads == 0, "The embeddings depth must be divisible by the number of heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale: float = self.d_head**-0.5

        self.has_flash_attn = hasattr(F, "scaled_dot_product_attention")
        if not self.has_flash_attn:
            logger.warning(
                "This program cannot run Flash Attention, for optimal computing, check your GPU driver and your PyTorch version"
            )

        self.q_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=bias)

        self._init_parameters()

        self.dropout = torch.nn.Dropout(dropout)

    def _init_parameters(self):
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """Split the heads and put them into a batch-first format."""
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.d_head)
        return tensor.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        """Merge heads and transpose back to batch-first format."""
        batch_size = tensor.shape[0]
        tensor = tensor.transpose(1, 2)
        return tensor.reshape(batch_size, -1, self.d_model).contiguous()

    def compute_flash_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combined_mask = None
            if attn_mask is not None:
                combined_mask = ~attn_mask  # (L, S) bool, True=attend
            if key_padding_mask is not None:
                kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
                combined_mask = kpm if combined_mask is None else combined_mask & kpm
            if combined_mask is not None:
                # Expand to explicit 4D so SDPA's CUDA kernel sees a contiguous last dim
                combined_mask = combined_mask.expand(q.shape[0], 1, q.shape[2], k.shape[2]).contiguous()
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=combined_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )

    def _compute_regular_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights.masked_fill_(~key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_weights)
        attn_output = attn_probs @ v

        return attn_output, attn_weights

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        return_weights: bool = False,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, tuple[Tensor, Tensor] | None]:

        if key is None and value is None:
            # self-attention
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)

            q = self._split_heads(q)
            k = self._split_heads(k)
            v = self._split_heads(v)

            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            present_key_value = (k, v) if use_cache else None
        else:
            if key is None or value is None:
                raise ValueError("Both key and value must be provided for cross-attention")

            q = self._split_heads(self.q_proj(query))
            k = self._split_heads(self.k_proj(key))
            v = self._split_heads(self.v_proj(value))
            present_key_value = None  # only caching decoder self-attn here

        use_flash_attn = self.has_flash_attn and not return_weights
        if use_flash_attn:
            attn_output = self.compute_flash_attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            attn_weights = None
        else:
            attn_output, attn_weights = self._compute_regular_attention(q, k, v, key_padding_mask, attn_mask)

        output = self.out_proj(self._merge_heads(attn_output))
        return output, (attn_weights if return_weights else None), present_key_value


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.activation = torch.nn.ReLU() if activation.lower() == "relu" else torch.nn.GELU()

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_ff),
            self.activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_ff, d_model),
        )

        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout_layers = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(3)])

    def forward(
        self,
        x: Tensor,
        encoder_output_key: Tensor,
        encoder_output_value: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        return_weights: bool = False,
        layer_past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ):
        attn_output, self_weights, present_key_value = self.self_attn(
            query=x,
            key=None,
            value=None,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_mask,
            return_weights=return_weights,
            past_key_value=layer_past,
            use_cache=use_cache,
        )

        x = self.norm_layers[0](x + self.dropout_layers[0](attn_output))

        attn_output, cross_weights, _ = self.cross_attn(
            query=x,
            key=encoder_output_key,
            value=encoder_output_value,
            key_padding_mask=memory_key_padding_mask,
            return_weights=return_weights,
        )

        x = self.norm_layers[1](x + self.dropout_layers[1](attn_output))
        x = self.norm_layers[2](x + self.dropout_layers[2](self.ffn(x)))

        if return_weights:
            return x, [self_weights, cross_weights], present_key_value
        return x, None, present_key_value


class DecoderStack(torch.nn.Module):
    def __init__(self, num_hidden_layers: int, d_model: int, dim_ff: int, num_heads: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                DecoderLayer(d_model=d_model, num_heads=num_heads, dim_ff=dim_ff, dropout=dropout)
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output_2D: Tensor,
        encoder_output_raw: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        return_weights: bool = False,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
        use_cache: bool = False,
    ):
        output = x
        all_weights: dict[str, list[Tensor]] = {"self_attn": [], "cross_attn": []}
        next_cache: list[tuple[Tensor, Tensor]] = []

        for i, dec_layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            output, weights, present = dec_layer(
                x=output,
                encoder_output_key=encoder_output_2D,
                encoder_output_value=encoder_output_raw,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                return_weights=return_weights,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            if return_weights:
                all_weights["self_attn"].append(weights[0])
                all_weights["cross_attn"].append(weights[1])
            if use_cache and present is not None:
                next_cache.append(present)

        if return_weights:
            return output, all_weights, (tuple(next_cache) if use_cache else None)
        return output, None, (tuple(next_cache) if use_cache else None)


class Decoder(torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        d_model: int,
        dim_ff: int,
        n_heads: int,
        max_seq_length: int,
        out_categories: int,
        dropout: float = 0.1,
    ):

        super().__init__()

        self.decoder = DecoderStack(
            num_hidden_layers=num_hidden_layers, d_model=d_model, dim_ff=dim_ff, num_heads=n_heads, dropout=dropout
        )
        self.embedding = torch.nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)
        self.position_encoding = PositionalEncoding1D(dim=d_model, len_max=max_seq_length)
        self.vocab_projection = torch.nn.Linear(in_features=d_model, out_features=out_categories)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        decoder_input: Tensor,
        encoder_output_2D: Tensor,
        encoder_output_raw: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        return_weights=False,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
        use_cache=False,
    ):
        decoder_input = self.embedding(decoder_input)
        start = past_key_values[0][0].size(2) if past_key_values is not None else 0
        decoder_input = self.position_encoding(decoder_input, start=start)

        output, weights, next_cache = self.decoder(
            x=decoder_input,
            encoder_output_2D=encoder_output_2D,
            encoder_output_raw=encoder_output_raw,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_weights=return_weights,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        output = self.dropout(output)

        predictions = self.vocab_projection(output)

        return output, predictions, weights, next_cache


class SMTOutput(CausalLMOutputWithCrossAttentions):
    """Output wrapper for the SMT"""


class SMTModelForCausalLM(transformers.PreTrainedModel, transformers.GenerationMixin):
    config_class = SMTConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SMTConfig):
        super().__init__(config)
        conv_next_stages = 3
        next_config = ConvNextConfig(
            num_channels=config.in_channels,  # type: ignore
            num_stages=conv_next_stages,  # type: ignore
            hidden_sizes=[config.d_model // 4, config.d_model // 2, config.d_model],  # type: ignore
            depths=[3, 3, 9],  # type: ignore
        )
        self.encoder = ConvNextModel(next_config)
        self.decoder = Decoder(
            num_hidden_layers=config.num_hidden_layers,
            d_model=config.d_model,
            dim_ff=config.dim_ff,
            n_heads=config.num_attn_heads,
            max_seq_length=config.maxlen,
            out_categories=config.vocab_size,
        )

        self.width_reduction = 2 ** (conv_next_stages + 1)
        self.height_reduction = 2 ** (conv_next_stages + 1)

        self.pos2D = PositionalEncoding2D(
            dim=config.d_model, h_max=config.maxh // self.height_reduction, w_max=config.maxw // self.width_reduction
        )

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.maxlen = int(config.maxlen)

    def _unwrap_cache(self, past_key_values):
        """Convert DynamicCache / EncoderDecoderCache to internal tuple-of-tuples."""
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        if past_key_values is None:
            return None
        if isinstance(past_key_values, EncoderDecoderCache):
            past_key_values = past_key_values.self_attention_cache
        if isinstance(past_key_values, DynamicCache):
            if past_key_values.get_seq_length() == 0:
                return None
            return tuple((layer.keys, layer.values) for layer in past_key_values.layers if layer.is_initialized)  # type: ignore[union-attr]
        return past_key_values  # already tuple format

    def _wrap_cache(self, next_cache):
        """Convert internal tuple-of-tuples to DynamicCache."""
        from transformers.cache_utils import DynamicCache

        if not next_cache:
            return None
        cache = DynamicCache()
        for i, (k, v) in enumerate(next_cache):
            cache.update(k, v, i)
        return cache

    def get_encoder(self):  # type: ignore[override]
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward_encoder(self, x: Tensor) -> Tensor:
        return self.encoder(pixel_values=x).last_hidden_state

    def forward_decoder(
        self, encoder_output, last_predictions, return_weights=False, past_key_values=None, use_cache=False
    ):
        b, _, _, _ = encoder_output.size()

        encoder_output_2D = self.pos2D(encoder_output)
        encoder_features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(0, 2, 1)
        encoder_features_2D = torch.flatten(encoder_output_2D, start_dim=2, end_dim=3).permute(0, 2, 1)
        token_lens = (last_predictions != self.config.pad_token_id).sum(dim=1).tolist()
        key_target_mask = self._generate_token_mask(token_lens, last_predictions.size(), device=last_predictions.device)
        causal_mask = self._generate_causal_mask(last_predictions.size(1), last_predictions.device)

        output, predictions, weights, next_cache = self.decoder(
            decoder_input=last_predictions,
            encoder_output_2D=encoder_features_2D,
            encoder_output_raw=encoder_features,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_target_mask,
            memory_key_padding_mask=None,
            return_weights=return_weights,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return SMTOutput(
            logits=predictions,
            hidden_states=output,
            attentions=None if weights is None else weights["self_attn"],
            cross_attentions=None if weights is None else weights["cross_attn"],
            past_key_values=next_cache,
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        encoder_outputs: BaseModelOutput | None = None,
        labels: Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You must provide `pixel_values` or `encoder_outputs`.")
            encoder_hidden = self.forward_encoder(pixel_values)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)  # type: ignore
        else:
            assert encoder_outputs.last_hidden_state is not None
            encoder_hidden = encoder_outputs.last_hidden_state.as_subclass(Tensor)

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        if decoder_input_ids is None:
            raise ValueError("You must provide `decoder_input_ids` (or `labels`).")

        if labels is not None:
            use_cache = False

        dec_out = self.forward_decoder(
            encoder_hidden,
            decoder_input_ids,
            return_weights=False,
            past_key_values=self._unwrap_cache(past_key_values),
            use_cache=bool(use_cache),
        )

        logits = dec_out.logits

        loss = None
        if labels is not None:
            loss = self.loss(logits.permute(0, 2, 1).contiguous(), labels)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=self._wrap_cache(dec_out.past_key_values),
            decoder_hidden_states=None,
            decoder_attentions=dec_out.attentions,
            cross_attentions=dec_out.cross_attentions,
            encoder_last_hidden_state=encoder_hidden,  # type: ignore
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        from transformers.cache_utils import DynamicCache

        if past_key_values is None:
            return None
        if isinstance(past_key_values, DynamicCache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        return tuple(
            (
                layer_k.index_select(0, beam_idx),
                layer_v.index_select(0, beam_idx),
            )
            for layer_k, layer_v in past_key_values
        )

    def _generate_token_mask(self, token_len, total_size, device):
        batch_size, len_mask = total_size
        mask = torch.zeros((batch_size, len_mask), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = True

        return mask

    def _generate_causal_mask(self, token_len, device):
        causal_mask = torch.triu(torch.ones(token_len, token_len, dtype=torch.bool, device=device), diagonal=1)
        return causal_mask

    def _shift_right(self, labels: Tensor) -> Tensor:
        pad_token_id = self.config.pad_token_id
        start_token_id = self.config.decoder_start_token_id
        if pad_token_id is None or start_token_id is None:
            raise ValueError("`pad_token_id` and `decoder_start_token_id` must be set in config.")

        shifted = labels.new_full(labels.shape, pad_token_id)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = start_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def prepare_inputs_for_generation(  # type: ignore[override]
        self,
        decoder_input_ids: Tensor,
        past_key_values=None,
        encoder_outputs: BaseModelOutput | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        if self._unwrap_cache(past_key_values) is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        model_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

        if encoder_outputs is None:
            model_inputs["pixel_values"] = kwargs.get("pixel_values")

        return model_inputs


@_build_model.register
def _(architecture_config: ArchitectureConfig) -> transformers.PreTrainedModel:
    params = architecture_config.model_dump(exclude={"kind"})
    return SMTModelForCausalLM(SMTConfig(**params))


@_build_preprocessor.register
def _(preprocessor_cfg: PreprocessorConfig, vocab: list[str]) -> BasePreprocessor:
    return Preprocessor.from_vocab(preprocessor_cfg, vocab)
