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
ARCH=llava; CUDA_VISIBLE_DEVICES=...; uv run scripts/train.py config/train.yaml --override architecture=$ARCH --override preprocessor_path=artifacts/preprocessors/$ARCH
```
