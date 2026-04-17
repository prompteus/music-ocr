import os
import pathlib
import typing
from typing import Any, Callable

import datasets
import hydra
import lightning
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import omegaconf
import pydantic
import torch
import typer
import wandb.sdk
import wandb.sdk.wandb_run
import yaml

import music_ocr.data
import music_ocr.kern
import music_ocr.model
import music_ocr.model.base
import music_ocr.tokenizer
import music_ocr.train_callbacks
import music_ocr.trainer

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)


class GenerativeEvalConfig(pydantic.BaseModel, extra="forbid"):
    max_new_tokens: int
    eval_n_examples: int


class TorchConfig(pydantic.BaseModel, extra="forbid"):
    num_threads: int
    float32_matmul_precision: str


class FormattingConfig(pydantic.BaseModel, extra="forbid"):
    convert_to: music_ocr.kern.KernFormat


class Config(pydantic.BaseModel, extra="forbid"):
    global_seed: int = 42
    torch: TorchConfig
    architecture: music_ocr.model.ArchitectureConfig
    scheduler: dict
    optimizer: dict
    trainer: dict
    preprocessor_path: pathlib.Path
    dataset: music_ocr.data.DatasetConfig
    formatting: FormattingConfig
    train_loader: dict
    valid_loader: dict
    gen_eval: GenerativeEvalConfig


def resolve_config(
    config_path: pathlib.Path,
    override: list[str] | None = None,
) -> tuple[omegaconf.DictConfig, Config]:
    with hydra.initialize_config_dir(str(config_path.parent.absolute()), version_base=None):
        omega_config = hydra.compose(config_name=config_path.stem, overrides=override)

    config_dict = omegaconf.OmegaConf.to_container(omega_config, resolve=True)
    config: Config = Config.model_validate(config_dict)
    return omega_config, config


@app.command()
def main(
    config_path: pathlib.Path = typer.Argument(..., help="Path to the training config file."),
    override: list[str] | None = typer.Option(
        None,
        "--override",
        help="Hydra config overrides to apply on top of the config file. Can be used multiple times.",
    ),
):
    omega_cfg, cfg = resolve_config(config_path, override)
    torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)
    torch.set_num_threads(cfg.torch.num_threads)
    lightning.seed_everything(cfg.global_seed)

    typer.secho("Loading preprocessor...", fg=typer.colors.CYAN)
    preprocessor = music_ocr.model.load_preprocessor(cfg.preprocessor_path)
    assert isinstance(preprocessor, music_ocr.model.base.Preprocessor)

    typer.secho("Verifying architecture config consistency with loaded tokenizer...", fg=typer.colors.CYAN)
    try:
        verified_paths = verify_model_config(cfg.architecture, preprocessor.tokenizer)
        typer.secho("Found the following verified architecture config paths:", fg=typer.colors.CYAN)
        for path in verified_paths:
            typer.secho(f"  - {path}", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho("Error occurred while verifying architecture config:", fg=typer.colors.RED)
        raise e

    typer.secho("Loading dataset...", fg=typer.colors.CYAN)
    ds = datasets.load_dataset(cfg.dataset.name)
    ds = ds.map(parse_kern_row, fn_kwargs={"txt_col": cfg.dataset.txt_col, "krn_format": cfg.formatting.convert_to})

    ds_train = music_ocr.data.OCRDataset(
        ds[cfg.dataset.train_split_name],
        img_col=cfg.dataset.img_col,
        txt_col=cfg.dataset.txt_col,
        preprocess=preprocessor.preprocess_one,
        pass_label_to_preprocess=True,
    )
    ds_valid = music_ocr.data.OCRDataset(
        ds[cfg.dataset.valid_split_name],
        img_col=cfg.dataset.img_col,
        txt_col=cfg.dataset.txt_col,
        preprocess=preprocessor.preprocess_one,
        pass_label_to_preprocess=True,
    )

    n_gen_eval = cfg.gen_eval.eval_n_examples
    _ds_valid_gen = music_ocr.data.OCRDataset(
        ds[cfg.dataset.valid_split_name],
        img_col=cfg.dataset.img_col,
        txt_col=cfg.dataset.txt_col,
        preprocess=preprocessor.preprocess_one,
        pass_label_to_preprocess=False,
    )
    ds_valid_gen = torch.utils.data.Subset(_ds_valid_gen, torch.randperm(len(_ds_valid_gen))[:n_gen_eval].tolist())

    def collate_batch(examples: list[tuple[dict, str]]) -> music_ocr.trainer.Batch:
        raw_inputs = [example[0] for example in examples]
        labels_str = [example[1] for example in examples]
        inputs = preprocessor.process_batch(raw_inputs)
        return music_ocr.trainer.Batch(inputs, labels_str)

    typer.secho("Building dataloader...", fg=typer.colors.CYAN)
    train_loader = torch.utils.data.DataLoader(ds_train, collate_fn=collate_batch, **cfg.train_loader)
    valid_loader = torch.utils.data.DataLoader(ds_valid, collate_fn=collate_batch, **cfg.valid_loader)
    valid_loader_gen = torch.utils.data.DataLoader(ds_valid_gen, collate_fn=collate_batch, **cfg.valid_loader)

    class DataLoaderMeta(typing.NamedTuple):
        prefix: str
        is_gen_eval: bool
        dataloader: torch.utils.data.DataLoader

    val_dataloaders = [
        DataLoaderMeta(prefix="valid", is_gen_eval=False, dataloader=valid_loader),
        DataLoaderMeta(prefix="valid", is_gen_eval=True, dataloader=valid_loader_gen),
    ]

    typer.secho("Building model...", fg=typer.colors.CYAN)
    lmodule = music_ocr.trainer.OCRLightning(
        architecture_config=cfg.architecture.model_dump(),
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        valid_loader_prefixes=[meta.prefix for meta in val_dataloaders],
        valid_loader_is_gen_eval=[meta.is_gen_eval for meta in val_dataloaders],
    )
    lmodule.set_tokenizer(preprocessor.tokenizer)

    typer.secho("Setting up logging and checkpointing...", fg=typer.colors.CYAN)
    wandb_logger = lightning.pytorch.loggers.WandbLogger(project="music-ocr", save_dir=".wandb/")
    run: wandb.sdk.wandb_run.Run = wandb_logger.experiment
    output_dir = pathlib.Path(f"./checkpoints/{run.project}/{run.name}/")
    os.makedirs(output_dir, exist_ok=True)
    checkpointing = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        monitor="valid/loss",
        filename="step@{step}-valid_loss@{valid/loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        mode="min",
    )
    gradnorm_logger = music_ocr.train_callbacks.GradientNormLogger(norm_type=2)
    lr_logger = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
        log_weight_decay=True,
    )

    typer.secho("Building trainer...", fg=typer.colors.CYAN)
    trainer = lightning.Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=[checkpointing, gradnorm_logger, lr_logger],
    )

    typer.secho("Saving preprocessor to checkpoint directory...", fg=typer.colors.CYAN)
    preprocessor.save(output_dir / "preprocessor")

    typer.secho("Saving config...", fg=typer.colors.CYAN)
    config_dict = cfg.model_dump(mode="json")
    config_yaml_resolved = yaml.dump(config_dict, sort_keys=False)
    config_yaml_orig = omegaconf.OmegaConf.to_yaml(omega_cfg)

    with open(output_dir / "config.yaml", "w") as f:
        f.write(config_yaml_orig)
    with open(output_dir / "config_resolved.yaml", "w") as f:
        f.write(config_yaml_resolved)

    # this makes it simple to filter runs according to config settings
    if hasattr(run.config, "update"):
        # without this check, wandb.config.update crashes when multiple GPUs are visible:
        # see: https://github.com/Lightning-AI/pytorch-lightning/discussions/13157
        run.config.update(config_dict)  # type: ignore[no-untyped-call]

    # this makes it easy to copy-paste config from wandb for reproduction
    wandb_logger.log_table(
        "config",
        columns=["config_yaml", "config_yaml_resolved"],
        data=[[config_yaml_orig, config_yaml_resolved]],
    )

    typer.secho("Training...", fg=typer.colors.CYAN)
    trainer.fit(
        lmodule,
        train_dataloaders=train_loader,
        val_dataloaders=[dataloader_meta.dataloader for dataloader_meta in val_dataloaders],
    )

    typer.secho("Exiting", fg=typer.colors.CYAN)


def parse_kern_row(row: dict[str, Any], txt_col: str, krn_format: music_ocr.kern.KernFormat) -> dict[str, str]:
    return {txt_col: " ".join(music_ocr.kern.parse_kern(row[txt_col], krn_format=krn_format))}


def verify_model_config(
    arch_cfg: music_ocr.model.ArchitectureConfig,
    tokenizer: music_ocr.tokenizer.Tokenizer,
) -> list[str]:
    """
    checks recursively for any key from these if it exists:
    - 'vocab_size' matches between the architecture config and the tokenizer, fills in from tokenizer if None in config
    - 'pad_token_id' matches with tokenizer.pad_token_id
    - 'eos_token_id' matches with tokenizer.eos_token_id
    - 'decoder_start_token_id' matches with tokenizer.bos_token_id
    - 'bos_token_id' matches with tokenizer.bos_token_id
    - 'image_token_id' matches with "<IMG_PATCH>" token id in tokenizer
    """
    img_patch_token_id: int | None
    img_start_token_id: int | None
    img_end_token_id: int | None
    try:
        img_patch_token_id = tokenizer.get_token_id("<IMG_PATCH>")
    except KeyError:
        img_patch_token_id = None
    try:
        img_start_token_id = tokenizer.get_token_id("<IMG_START>")
    except KeyError:
        img_start_token_id = None
    try:
        img_end_token_id = tokenizer.get_token_id("<IMG_END>")
    except KeyError:
        img_end_token_id = None

    key_to_expected: dict[str, int | None] = {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "decoder_start_token_id": tokenizer.bos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "image_token_id": img_patch_token_id,
        "image_token_index": img_patch_token_id,
        "image_start_token_id": img_start_token_id,
        "image_end_token_id": img_end_token_id,
    }
    verified_paths: list[str] = []

    def process_leaf(key: str, value: Any, path: str, set_value: typing.Callable[[int], None]) -> None:
        if key not in key_to_expected:
            return
        expected = key_to_expected[key]
        if expected is None:
            raise ValueError(f"Tokenizer does not define an id required for {path}: expected value is None")
        if key == "vocab_size" and value is None:
            set_value(expected)
            verified_paths.append(path)
            return
        if value != expected:
            raise ValueError(f"Model config mismatch at {path}: found {value!r}, expected {expected!r}")
        verified_paths.append(path)

    def set_attr(obj: object, name: str) -> Callable[[int], None]:
        def _set(value: int) -> None:
            setattr(obj, name, value)

        return _set

    def set_dict_item(obj: dict[Any, Any], key: Any) -> Callable[[int], None]:
        def _set(value: int) -> None:
            obj[key] = value

        return _set

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, pydantic.BaseModel):
            for field_name in type(obj).model_fields:
                value = getattr(obj, field_name)
                field_path = f"{path}.{field_name}"
                process_leaf(field_name, value, field_path, set_attr(obj, field_name))
                walk(value, field_path)

            extras = getattr(obj, "__pydantic_extra__", None) or {}
            for extra_key in list(extras.keys()):
                value = extras[extra_key]
                extra_path = f"{path}.{extra_key}"
                process_leaf(extra_key, value, extra_path, set_dict_item(extras, extra_key))
                walk(value, extra_path)
            return

        if isinstance(obj, dict):
            for key in list(obj.keys()):
                value = obj[key]
                item_path = f"{path}.{key}"
                process_leaf(str(key), value, item_path, set_dict_item(obj, key))
                walk(value, item_path)
            return

    walk(arch_cfg, "")
    return [path.lstrip(".") for path in verified_paths]


if __name__ == "__main__":
    app()
