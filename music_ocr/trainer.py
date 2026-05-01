from typing import NamedTuple, cast

import lightning
import tensordict
import torch
import torchmetrics.functional.text
import transformers
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

import music_ocr.model
import music_ocr.optim
import music_ocr.tokenizer


class Batch(NamedTuple):
    inputs: tensordict.TensorDict | dict
    labels_str: list[str]


class OCRLightning(lightning.LightningModule):
    def __init__(
        self,
        architecture_config: dict,
        optimizer_config: dict,
        scheduler_config: dict | None = None,
        valid_loader_prefixes: list[str] | None = None,
        valid_loader_is_gen_eval: list[bool] | None = None,
        gen_eval_max_new_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.model = music_ocr.model.build_model(architecture_config)
        self.valid_loader_prefixes = valid_loader_prefixes
        self.valid_loader_is_gen_eval = valid_loader_is_gen_eval
        self.gen_eval_max_new_tokens = gen_eval_max_new_tokens
        self.tokenizer: music_ocr.tokenizer.Tokenizer | None = None

    def configure_optimizers(self):
        optimizer = music_ocr.optim.build_optimizer(self.model.parameters(), self.optimizer_config)
        if self.scheduler_config is None:
            return optimizer
        scheduler = transformers.get_scheduler(optimizer=optimizer, **self.scheduler_config)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    def set_tokenizer(self, tokenizer: music_ocr.tokenizer.Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.model.generation_config.eos_token_id = tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = tokenizer.pad_token_id

    def forward(self, batch: tensordict.TensorDict | dict) -> Tensor:
        return self.model(**batch)  # type: ignore

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        bs = len(batch.labels_str)
        output: CausalLMOutputWithPast = self.model(**batch.inputs)  # type: ignore
        assert output.loss is not None
        if not torch.isfinite(output.loss):
            raise RuntimeError("Non-finite loss detected during training step")
        self.log("train/loss", output.loss.item(), batch_size=bs)
        return output.loss

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int) -> None:
        bs = len(batch.labels_str)
        if self.valid_loader_is_gen_eval and self.valid_loader_is_gen_eval[dataloader_idx]:
            metrics = self._validation_step_gen_eval(batch)
        else:
            metrics = self._validation_step_loss_eval(batch)
        prefix = self._get_loader_prefix(dataloader_idx)
        self.log_dict({f"{prefix}/{k}": v for k, v in metrics.items()}, batch_size=bs, add_dataloader_idx=False)

    def _validation_step_loss_eval(self, batch: Batch) -> dict[str, float]:
        output: CausalLMOutputWithPast = self.model(**batch.inputs)  # type: ignore
        assert output.loss is not None
        return {"loss": output.loss.item()}

    def _validation_step_gen_eval(self, batch: Batch) -> dict[str, float]:
        """
        Make sure that the batch here is prepared correctly for generation evaluation,
        i.e. that the input_ids only contain the "prefix" and that the labels only the "suffix" of the whole sequence.
        """
        assert self.tokenizer is not None, "Tokenizer must be provided for generation evaluation"
        max_new_tokens = self.gen_eval_max_new_tokens if not self.trainer.sanity_checking else 10
        preds = self.model.generate(**batch.inputs, max_new_tokens=max_new_tokens)  # type: ignore
        assert isinstance(preds, torch.Tensor)
        preds_str = self.tokenizer.decode_batch(preds.tolist(), skip_special_tokens=True, stop_on_eos=True)
        ser = [p == t for p, t in zip(preds_str, batch.labels_str)].count(False) / len(batch.labels_str)
        wer = torchmetrics.functional.text.word_error_rate(preds_str, batch.labels_str).item()
        return {"ser": ser, "wer": wer}

    def _get_loader_prefix(self, dataloader_idx: int) -> str:
        if self.valid_loader_prefixes:
            return self.valid_loader_prefixes[dataloader_idx]
        return f"valid_{dataloader_idx}"
