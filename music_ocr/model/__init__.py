import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Union

import pydantic
import transformers

from music_ocr.model.base import (
    Preprocessor as BasePreprocessor,
)
from music_ocr.model.base import (
    build_model,
    build_preprocessor,
)

# Import all model modules so subclasses are registered
from . import (
    llava,  # noqa: F401
)
from .base import ArchitectureConfig as BaseArch
from .base import PreprocessorConfig as BasePrep

if TYPE_CHECKING:
    ArchitectureConfig = BaseArch
    PreprocessorConfig = BasePrep
else:
    ArchUnion = Union[tuple(BaseArch.__subclasses__())]
    PrepUnion = Union[tuple(BasePrep.__subclasses__())]
    ArchitectureConfig = Annotated[ArchUnion, pydantic.Field(discriminator="kind")]
    PreprocessorConfig = Annotated[PrepUnion, pydantic.Field(discriminator="kind")]


@build_model.register
def _(architecture_config: dict) -> transformers.PreTrainedModel:
    typed: ArchitectureConfig
    typed = pydantic.TypeAdapter(ArchitectureConfig).validate_python(architecture_config)
    return build_model(typed)


@build_preprocessor.register
def _(preprocessor_cfg: dict, vocab: list[str]) -> BasePreprocessor:
    typed: PreprocessorConfig
    typed = pydantic.TypeAdapter(PreprocessorConfig).validate_python(preprocessor_cfg)
    return build_preprocessor(typed, vocab)


def load_preprocessor(path: Path | str) -> BasePreprocessor:
    """Load a preprocessor that was saved with Preprocessor.save()."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    typed: PreprocessorConfig
    typed = pydantic.TypeAdapter(PreprocessorConfig).validate_python(config)
    return typed.load(path)


__all__ = ["ArchitectureConfig", "PreprocessorConfig", "build_model", "build_preprocessor", "load_preprocessor"]
