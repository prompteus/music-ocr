# Music OCR

## Setup

```shell
git clone ...
cd ...
uv sync
uv pip install -e .
pre-commit install
```

## Build a Preprocessor
```shell
# Real dataset only (default - grandstaff_ekern):
ARCH=llava; uv run scripts/build_preprocessor.py config/build_preprocessor.yaml --override preprocessor=$ARCH artifacts/preprocessors/$ARCH
# Both datasets (default + synth_omr_500k):
ARCH=llava; uv run scripts/build_preprocessor.py config/build_preprocessor.yaml --override preprocessor=$ARCH artifacts/preprocessors/$ARCH \
  --override '+dataset@datasets.synthetic=synth_omr_500k'
```

## Run Training
```shell
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ARCH=llava  # smt, qwen25vl, llava, glm_ocr
CUDA_VISIBLE_DEVICES=...

# Stage 1: Train on synthetic data
uv run scripts/train.py config/train.yaml \
    --override architecture=$ARCH \
    --override preprocessor_path=artifacts/preprocessors/$ARCH \
    --override dataset=synth_omr_500k

# Stage 2: Fine-tune on real data (loads only weights, resets optimizer + epochs)
# Replace <run_name> with the wandb run name from stage 1 (shown in terminal on startup)
CKPT=/path/to/checkpoints/music-ocr/<run_name>/step@XXXX-valid_loss@X.XXXX.ckpt
uv run scripts/train.py config/train.yaml \
    --override architecture=$ARCH \
    --override preprocessor_path=artifacts/preprocessors/$ARCH \
    --finetune-from $CKPT

# Resume an interrupted run (restores weights + optimizer + epoch)
uv run scripts/train.py config/train.yaml \
    --override architecture=$ARCH \
    --override preprocessor_path=artifacts/preprocessors/$ARCH \
    --ckpt-path $CKPT
```


