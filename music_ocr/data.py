from collections.abc import Callable
from typing import Any

import datasets
import PIL.Image
import pydantic
import torch


class DatasetConfig(pydantic.BaseModel, extra="forbid"):
    name: str
    img_col: str
    txt_col: str
    train_split_name: str
    valid_split_name: str


class OCRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: datasets.Dataset,
        img_col: str,
        txt_col: str,
        preprocess: Callable[[PIL.Image.Image, str | None], Any],
        pass_label_to_preprocess: bool,
    ) -> None:
        super().__init__()
        self.data = data
        self.img_col = img_col
        self.txt_col = txt_col
        self.preprocess = preprocess
        self.pass_label_to_preprocess = pass_label_to_preprocess

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        image = self.data[self.img_col][index]
        label = self.data[self.txt_col][index]
        preprocess_label = label if self.pass_label_to_preprocess else None
        batch = self.preprocess(image, preprocess_label)
        return batch, label
