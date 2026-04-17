import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import PIL.Image
import pydantic
import tensordict
import transformers

import music_ocr.tokenizer


@runtime_checkable
class Preprocessor(Protocol):
    @property
    def tokenizer(self) -> music_ocr.tokenizer.Tokenizer: ...

    def save(self, path: Path) -> None: ...

    def preprocess_one(
        self,
        image: PIL.Image.Image,
        label: str | None,
    ) -> Any: ...

    def process_batch(self, batch: list[Any]) -> tensordict.TensorDict: ...

    def __call__(self, image: list[PIL.Image.Image], label: list[str] | None = None) -> tensordict.TensorDict: ...


class PreprocessorConfig(pydantic.BaseModel, ABC, extra="forbid"):
    """Base class for all preprocessor configurations."""

    kind: str

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Preprocessor: ...


class ArchitectureConfig(pydantic.BaseModel, ABC, extra="allow"):
    """Base class for all architecture configurations."""

    kind: str


@functools.singledispatch
def build_model(architecture_config: Any) -> transformers.PreTrainedModel:
    raise NotImplementedError(f"No build_model registered for {type(architecture_config)!r}")


@functools.singledispatch
def build_preprocessor(preprocessor_cfg: Any, vocab: list[str]) -> Preprocessor:
    raise NotImplementedError(f"No build_preprocessor registered for {type(preprocessor_cfg)!r}")
