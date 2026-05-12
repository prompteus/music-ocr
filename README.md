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
ARCH=llava; uv run scripts/build_preprocessor.py config/build_preprocessor.yaml --override preprocessor=$ARCH artifacts/preprocessors/$ARCH
```

## Run Training
```shell
export ARCH=llava  # smt, qwen25vl, llava, glm_ocr

# Stage 1: Train on synthetic data
uv run scripts/train.py config/train.yaml \
    --override architecture=$ARCH \
    --override preprocessor_path=artifacts/preprocessors/$ARCH

# Stage 2: Fine-tune on real data (loads only weights)
uv run scripts/train.py config/train.yaml \
    --override architecture=$ARCH \
    --override preprocessor_path=artifacts/preprocessors/$ARCH \
    --load-weights-ckpt </path/to/checkpoint.ckpt>

# Resume an interrupted run (restores weights + optimizer state + epoch)
uv run scripts/train.py config/train.yaml \
    --override architecture=$ARCH \
    --override preprocessor_path=artifacts/preprocessors/$ARCH \
    --resume-ckpt </path/to/checkpoint.ckpt>
```


