from pathlib import Path
from typing import Any, Literal, Self, cast

import PIL.Image
import tensordict
import torch
import torchvision.transforms.functional
import transformers
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

import music_ocr.tokenizer
import music_ocr.utils
from music_ocr.model.base import ArchitectureConfig as BaseArchitectureConfig
from music_ocr.model.base import Preprocessor as BasePreprocessor
from music_ocr.model.base import PreprocessorConfig as BasePreprocessorConfig
from music_ocr.model.base import build_model as _build_model
from music_ocr.model.base import build_preprocessor as _build_preprocessor


class ArchitectureConfig(BaseArchitectureConfig):
    kind: Literal["qwen25vl"] = "qwen25vl"
    # extra fields are passed to Qwen2_5_VLConfig


class PreprocessorConfig(BasePreprocessorConfig):
    kind: Literal["qwen25vl"] = "qwen25vl"
    tokenizer: music_ocr.tokenizer.SpaceTokenizerConfig
    image_processor: dict[str, Any]
    pad_to_multiple_of: int | None = 16

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        return Preprocessor.load(path)


class Preprocessor:
    class SingleInput(tensordict.TensorClass):
        input_ids: torch.Tensor
        image: torch.Tensor
        labels: torch.Tensor | None = None

    class Inputs(tensordict.TensorClass):
        input_ids: torch.Tensor
        attention_mask: torch.Tensor
        mm_token_type_ids: torch.Tensor
        pixel_values: torch.Tensor
        image_grid_thw: torch.Tensor
        second_per_grid_ts: torch.Tensor
        labels: torch.Tensor | None = None

    def __init__(self, config: PreprocessorConfig, tokenizer: music_ocr.tokenizer.SpaceTokenizer) -> None:
        assert isinstance(tokenizer, music_ocr.tokenizer.Tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.get_token_id("<SEP>")
        self.img_token_id = tokenizer.get_token_id("<IMG_PATCH>")
        self.image_processor = transformers.Qwen2VLImageProcessor(**config.image_processor)

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

    def preprocess_one(
        self,
        image: PIL.Image.Image,
        label: str | None,
    ) -> SingleInput:
        image = image.convert("L")
        factor = self.image_processor.patch_size * self.image_processor.merge_size
        resized_h, resized_w = smart_resize(
            image.height,
            image.width,
            factor=factor,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )
        image = image.resize((resized_w, resized_h), self.image_processor.resample)
        image_tensor = torchvision.transforms.functional.pil_to_tensor(image)
        n_img_tokens = (
            (resized_h // self.image_processor.patch_size)
            * (resized_w // self.image_processor.patch_size)
            // (self.image_processor.merge_size**2)
        )
        prefix_ids = [self.bos_token_id] + [self.img_token_id] * n_img_tokens + [self.sep_token_id]
        if label is None:
            return self.SingleInput(input_ids=prefix_ids, image=image_tensor)
        label_ids = self.tokenizer.encode(label) + [self.eos_token_id]
        input_ids = prefix_ids + label_ids
        labels = [-100] * len(prefix_ids) + label_ids
        return self.SingleInput(
            input_ids=input_ids,
            image=image_tensor,
            labels=labels,
        )

    def process_batch(self, batch: list[SingleInput]) -> tensordict.TensorDict:
        images = self.image_processor(images=[item.image for item in batch], return_tensors="pt")
        input_ids = music_ocr.utils.pad_sequence(
            [item.input_ids for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )
        collated = self.Inputs(
            input_ids=input_ids,
            attention_mask=(input_ids != self.pad_token_id),
            mm_token_type_ids=(input_ids == self.img_token_id),
            pixel_values=images.pixel_values,
            image_grid_thw=images.image_grid_thw,
            second_per_grid_ts=torch.ones(len(batch), dtype=torch.long),
            labels=None,
        )
        if batch[0].labels is not None:
            assert all(item.labels is not None for item in batch), "Either all or none of the examples must have labels"
            collated.labels = music_ocr.utils.pad_sequence(
                [item.labels for item in batch],  # type: ignore
                batch_first=True,
                padding_value=-100,
                padding_side="left",
                pad_to_multiple_of=self.config.pad_to_multiple_of,
            )
        return cast(tensordict.TensorDict, collated.to_tensordict())

    def __call__(
        self, image: list[PIL.Image.Image], label: list[str] | list[None] | None = None
    ) -> tensordict.TensorDict:
        if label is None:
            label = [None] * len(image)
        return self.process_batch([self.preprocess_one(img, lbl) for img, lbl in zip(image, label)])


@_build_model.register
def _(architecture_config: ArchitectureConfig) -> transformers.PreTrainedModel:
    params = architecture_config.model_dump(exclude={"kind"})
    return transformers.Qwen2_5_VLForConditionalGeneration(transformers.Qwen2_5_VLConfig(**params))


@_build_preprocessor.register
def _(preprocessor_cfg: PreprocessorConfig, vocab: list[str]) -> BasePreprocessor:
    return Preprocessor.from_vocab(preprocessor_cfg, vocab)
