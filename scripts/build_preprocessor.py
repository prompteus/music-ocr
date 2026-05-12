import pathlib

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
    datasets: dict[str, music_ocr.data.DatasetConfig]
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

    vocab_train: set[str] = set()
    vocab: set[str] = set()
    vocab_per_dataset: dict[str, set[str]] = {}

    dataset_items = list(cfg.datasets.items())
    for ds_idx, (ds_key, dataset_cfg) in enumerate(dataset_items):
        typer.secho(
            f"[{ds_idx + 1}/{len(dataset_items)}] Loading dataset {ds_key!r} ({dataset_cfg.hf_handle!r})...",
            fg=typer.colors.CYAN,
        )
        ds: datasets.DatasetDict = datasets.load_dataset(dataset_cfg.hf_handle)

        if dataset_cfg.split_name is not None and dataset_cfg.split_name not in ds:
            raise ValueError(
                f"datasets[{ds_key!r}].split_name={dataset_cfg.split_name!r} not found in dataset"
                f" {dataset_cfg.hf_handle!r}. Available splits: {list(ds.keys())}"
            )

        krn_format = cfg.formatting.convert_to
        txt_col = dataset_cfg.txt_col

        typer.secho(f"[{ds_idx + 1}/{len(dataset_items)}] Parsing texts...", fg=typer.colors.CYAN)
        ds = ds.select_columns([dataset_cfg.txt_col])
        ds.set_transform(
            lambda batch: {
                txt_col: [" ".join(music_ocr.kern.parse_kern(t, krn_format=krn_format)) for t in batch[txt_col]]
            }
        )

        typer.secho(f"[{ds_idx + 1}/{len(dataset_items)}] Adding to vocabulary...", fg=typer.colors.CYAN)
        if dataset_cfg.split_name is not None:
            for text in ds[dataset_cfg.split_name][dataset_cfg.txt_col]:
                vocab_train.update(str(text).split(" "))

        dataset_vocab: set[str] = set()
        for split in ds.keys():
            for text in ds[split][dataset_cfg.txt_col]:
                dataset_vocab.update(str(text).split(" "))
                vocab.update(str(text).split(" "))
        vocab_per_dataset[ds_key] = dataset_vocab

    typer.secho(f"Found {len(vocab)} unique tokens across {len(dataset_items)} datasets.", fg=typer.colors.CYAN)
    datasets_without_train_split = sorted(key for key, ds_cfg in cfg.datasets.items() if ds_cfg.split_name is None)
    if datasets_without_train_split:
        typer.secho(
            f"Note: datasets {datasets_without_train_split} have no train split (split_name is null)"
            f" and are not counted towards vocab_train.",
            fg=typer.colors.CYAN,
        )
    if len(vocab - vocab_train) > 0:
        typer.secho(
            f"Warning: There are {len(vocab - vocab_train)} tokens in the whole dataset combination"
            f" that do not appear in any training split. These tokens will be included in the"
            f" tokenizer vocabulary, but the model will not see them during training.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    if len(vocab_per_dataset) > 1:
        names = list(vocab_per_dataset.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]
                only_in_1 = vocab_per_dataset[name1] - vocab_per_dataset[name2]
                only_in_2 = vocab_per_dataset[name2] - vocab_per_dataset[name1]
                if only_in_1 or only_in_2:
                    typer.secho(
                        f"\nVocabulary differences between {name1!r} and {name2!r}:",
                        fg=typer.colors.MAGENTA,
                    )
                    if only_in_1:
                        typer.secho(f"  Only in {name1!r} ({len(only_in_1)} tokens):", fg=typer.colors.MAGENTA)
                        typer.secho(f"    {sorted(only_in_1)}", fg=typer.colors.WHITE)
                    if only_in_2:
                        typer.secho(f"  Only in {name2!r} ({len(only_in_2)} tokens):", fg=typer.colors.MAGENTA)
                        typer.secho(f"    {sorted(only_in_2)}", fg=typer.colors.WHITE)
                else:
                    typer.secho(
                        f"\nNo vocabulary differences between {name1!r} and {name2!r}.",
                        fg=typer.colors.GREEN,
                    )

    typer.secho("Building preprocessor...", fg=typer.colors.CYAN)
    vocab_list = sorted(vocab)
    preprocessor = music_ocr.model.build_preprocessor(cfg.preprocessor, vocab=vocab_list)

    typer.secho(f"Saving preprocessor to {output_path} ...", fg=typer.colors.CYAN)
    assert isinstance(preprocessor, music_ocr.model.base.Preprocessor)
    output_path.mkdir(parents=True, exist_ok=True)
    preprocessor.save(output_path)


if __name__ == "__main__":
    app()
