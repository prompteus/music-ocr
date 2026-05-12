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
# Both datasets (default + synthetic_omr_500k):
ARCH=llava; uv run scripts/build_preprocessor.py config/build_preprocessor.yaml --override preprocessor=$ARCH artifacts/preprocessors/$ARCH \
  --override '+dataset@datasets.synthetic=synthetic_omr_500k'
```

## Run Training
```shell
ARCH=llava; CUDA_VISIBLE_DEVICES=...; uv run scripts/train.py config/train.yaml --override architecture=$ARCH --override preprocessor_path=artifacts/preprocessors/$ARCH

# TODO init with pretrained weights from synth training + continue training on real data
```


