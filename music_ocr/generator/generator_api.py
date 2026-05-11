from typing import Tuple, Optional

from .utils.kern_to_png import KernToPng
from .utils.kern_generator import KernGenerator
from .utils.kern_annotation_parser import convert_kern_to_annotated
from .dataset import generate_hf_dataset
from .hf import upload_to_hf


class Generator:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.kg = KernGenerator() if seed is None else KernGenerator(seed)
        self.kern_2_png = KernToPng()

    def generateStaff(self, num_measures: int = 4, *, single_line: bool = True) -> Tuple[str, str]:
        kern: str = str(self.kg.generate(num_measures=num_measures))
        image_path: str = self.kern_2_png.render_to_image(kern, single_line=single_line)
        annotated_kern: str = convert_kern_to_annotated(kern)
        return image_path, annotated_kern

    @staticmethod
    def generateDataset(
        output_dir: str,
        train_samples: int = 500,
        val_samples: int = 50,
        num_workers: int = 4,
    ) -> None:
        generate_hf_dataset(
            train_samples=train_samples,
            val_samples=val_samples,
            output_dir=output_dir,
            num_workers=num_workers,
        )

    @staticmethod
    def uploadDataset(dataset_path: str, repo_id: str) -> None:
        upload_to_hf(dataset_path=dataset_path, repo_id=repo_id)
