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
    train_dataset: music_ocr.data.DatasetConfig
    valid_datasets: dict[str, music_ocr.data.DatasetConfig]
    formatting: FormattingConfig
    train_loader: dict
    valid_loader: dict
    gen_eval: GenerativeEvalConfig
    checkpointing: list[dict]


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
    resume_ckpt: pathlib.Path | None = typer.Option(
        None,
        help="Resume full training state (weights + optimizer + epoch) from a Lightning checkpoint.",
    ),
    load_weights_ckpt: pathlib.Path | None = typer.Option(
        None,
        help=(
            "Load ONLY model weights from a Lightning checkpoint, resetting optimizer and epoch counter. "
            "Use this to fine-tune on a new dataset (e.g. synth -> real) after a previous training run."
        ),
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
    train_ds_raw = datasets.load_dataset(cfg.train_dataset.hf_handle, split=cfg.train_dataset.split_name)
    train_ds_raw = train_ds_raw.map(
        parse_kern_row,
        fn_kwargs={"txt_col": cfg.train_dataset.txt_col, "krn_format": cfg.formatting.convert_to},
    )

    ds_train = music_ocr.data.OCRDataset(
        train_ds_raw,
        img_col=cfg.train_dataset.img_col,
        txt_col=cfg.train_dataset.txt_col,
        preprocess=preprocessor.preprocess_one,
        pass_label_to_preprocess=True,
    )

    def collate_batch(examples: list[tuple[dict, str]]) -> music_ocr.trainer.Batch:
        raw_inputs = [example[0] for example in examples]
        labels_str = [example[1] for example in examples]
        inputs = preprocessor.process_batch(raw_inputs)
        return music_ocr.trainer.Batch(inputs, labels_str)

    class DataLoaderMeta(typing.NamedTuple):
        prefix: str
        is_gen_eval: bool
        dataloader: torch.utils.data.DataLoader

    typer.secho("Building dataloaders...", fg=typer.colors.CYAN)
    train_loader = torch.utils.data.DataLoader(ds_train, collate_fn=collate_batch, **cfg.train_loader)

    val_dataloaders: list[DataLoaderMeta] = []
    n_gen_eval = cfg.gen_eval.eval_n_examples
    for ds_key, ds_config in cfg.valid_datasets.items():
        typer.secho(f"  Loading validation dataset '{ds_key}' ({ds_config.hf_handle})...", fg=typer.colors.CYAN)
        valid_ds_raw = datasets.load_dataset(ds_config.hf_handle, split=ds_config.split_name)
        valid_ds_raw = valid_ds_raw.map(
            parse_kern_row,
            fn_kwargs={"txt_col": ds_config.txt_col, "krn_format": cfg.formatting.convert_to},
        )
        ds_valid = music_ocr.data.OCRDataset(
            valid_ds_raw,
            img_col=ds_config.img_col,
            txt_col=ds_config.txt_col,
            preprocess=preprocessor.preprocess_one,
            pass_label_to_preprocess=True,
        )
        _ds_valid_gen = music_ocr.data.OCRDataset(
            valid_ds_raw,
            img_col=ds_config.img_col,
            txt_col=ds_config.txt_col,
            preprocess=preprocessor.preprocess_one,
            pass_label_to_preprocess=False,
        )
        ds_valid_gen = torch.utils.data.Subset(_ds_valid_gen, torch.randperm(len(_ds_valid_gen))[:n_gen_eval].tolist())
        val_dataloaders.append(
            DataLoaderMeta(
                prefix=ds_key,
                is_gen_eval=False,
                dataloader=torch.utils.data.DataLoader(ds_valid, collate_fn=collate_batch, **cfg.valid_loader),
            )
        )
        val_dataloaders.append(
            DataLoaderMeta(
                prefix=ds_key,
                is_gen_eval=True,
                dataloader=torch.utils.data.DataLoader(ds_valid_gen, collate_fn=collate_batch, **cfg.valid_loader),
            )
        )

    typer.secho("Building model...", fg=typer.colors.CYAN)
    lmodule = music_ocr.trainer.OCRLightning(
        architecture_config=cfg.architecture.model_dump(),
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        valid_loader_prefixes=[meta.prefix for meta in val_dataloaders],
        valid_loader_is_gen_eval=[meta.is_gen_eval for meta in val_dataloaders],
    )
    lmodule.set_tokenizer(preprocessor.tokenizer)

    if load_weights_ckpt is not None:
        typer.secho(
            f"Loading model weights from '{load_weights_ckpt}' (fine-tune mode, optimizer state discarded)...",
            fg=typer.colors.CYAN,
        )
        ckpt = torch.load(load_weights_ckpt, map_location="cpu", weights_only=False)
        # Lightning checkpoints store model weights under 'state_dict' with a 'model.' prefix
        state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        missing, unexpected = lmodule.model.load_state_dict(state_dict, strict=True)
        if missing:
            typer.secho(f"  Warning: missing keys in checkpoint: {missing}", fg=typer.colors.YELLOW)
        if unexpected:
            typer.secho(f"  Warning: unexpected keys in checkpoint: {unexpected}", fg=typer.colors.YELLOW)
        typer.secho("  Weights loaded successfully.", fg=typer.colors.GREEN)

    typer.secho("Setting up logging and checkpointing...", fg=typer.colors.CYAN)
    wandb_logger = lightning.pytorch.loggers.WandbLogger(project="music-ocr", save_dir=".wandb/")
    run: wandb.sdk.wandb_run.Run = wandb_logger.experiment
    output_dir = pathlib.Path(f"./checkpoints/{run.project}/{run.name}/")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_callbacks = [
        lightning.pytorch.callbacks.ModelCheckpoint(dirpath=output_dir, auto_insert_metric_name=False, **ckpt_cfg)
        for ckpt_cfg in cfg.checkpointing
    ]
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
        callbacks=[*checkpoint_callbacks, gradnorm_logger, lr_logger],
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
        ckpt_path=str(resume_ckpt) if resume_ckpt is not None else None,
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
