import pathlib
from typing import Any

import datasets
import hydra
import omegaconf
import pydantic
import typer

import music_ocr.data
import music_ocr.kern
import music_ocr.model
import music_ocr.model.base

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)


class FormattingConfig(pydantic.BaseModel, extra="forbid"):
    convert_to: music_ocr.kern.KernFormat


class Config(pydantic.BaseModel, extra="forbid"):
    dataset: music_ocr.data.DatasetConfig
    formatting: FormattingConfig
    preprocessor: music_ocr.model.PreprocessorConfig


def resolve_config(
    config_path: pathlib.Path,
    override: list[str] | None = None,
) -> Config:
    with hydra.initialize_config_dir(str(config_path.parent.absolute()), version_base=None):
        omega_config = hydra.compose(config_name=config_path.stem, overrides=override)
    config_dict = omegaconf.OmegaConf.to_container(omega_config, resolve=True)
    config: Config = Config.model_validate(config_dict)
    return config


@app.command()
def main(
    config_path: pathlib.Path = typer.Argument(..., help="Path to the training config file."),
    output_path: pathlib.Path = typer.Argument(..., help="Directory where the preprocessor will be saved."),
    override: list[str] | None = typer.Option(
        None,
        "--override",
        help="Hydra config overrides. Can be used multiple times.",
    ),
) -> None:
    typer.secho(f"Loading config from '{config_path}'...", fg=typer.colors.CYAN)
    cfg = resolve_config(pathlib.Path(config_path), override)

    typer.secho("Loading dataset...", fg=typer.colors.CYAN)
    ds: datasets.DatasetDict = datasets.load_dataset(cfg.dataset.name)
    if cfg.dataset.train_split_name not in ds:
        raise ValueError(
            f"Train split {cfg.dataset.train_split_name!r} not found in dataset {cfg.dataset.name!r}. Available splits: {list(ds.keys())}"
        )

    typer.secho("Parsing texts...", fg=typer.colors.CYAN)
    ds = ds.map(parse_kern_row, fn_kwargs={"txt_col": cfg.dataset.txt_col, "krn_format": cfg.formatting.convert_to})

    typer.secho("Building vocabulary...", fg=typer.colors.CYAN)
    vocab_train: set[str] = set()
    vocab: set[str] = set()
    for text in ds[cfg.dataset.train_split_name][cfg.dataset.txt_col]:
        vocab_train.update(str(text).split(" "))

    for split in ds.keys():
        for text in ds[split][cfg.dataset.txt_col]:
            vocab.update(str(text).split(" "))

    typer.secho(f"Found {len(vocab)} unique tokens in the dataset.", fg=typer.colors.CYAN)
    if len(vocab - vocab_train) > 0:
        typer.secho(
            f"Warning: There are {len(vocab - vocab_train)} tokens in the whole dataset that do not appear in the training split. "
            f"These tokens will be included in the tokenizer vocabulary, but the model will not see them during training.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    typer.secho("Building preprocessor...", fg=typer.colors.CYAN)
    vocab_list = sorted(vocab)
    preprocessor = music_ocr.model.build_preprocessor(cfg.preprocessor, vocab=vocab_list)

    typer.secho(f"Saving preprocessor to {output_path} ...", fg=typer.colors.CYAN)
    assert isinstance(preprocessor, music_ocr.model.base.Preprocessor)
    output_path.mkdir(parents=True, exist_ok=True)
    preprocessor.save(output_path)


def parse_kern_row(row: dict[str, Any], txt_col: str, krn_format: music_ocr.kern.KernFormat) -> dict[str, str]:
    return {txt_col: " ".join(music_ocr.kern.parse_kern(row[txt_col], krn_format=krn_format))}


if __name__ == "__main__":
    app()
