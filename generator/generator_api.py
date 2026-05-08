from typing import Tuple, Optional

from generator.utils.kern_to_png import KernToPng
from generator.utils.kern_generator import KernGenerator
from generator.utils.kern_annotation_parser import convert_kern_to_annotated


class Generator:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.kg = KernGenerator() if seed is None else KernGenerator(seed)
        self.kern_2_png = KernToPng()

    def generateStaff(self, num_measures: int = 4, *, single_line: bool = True) -> Tuple[str, str]:
        kern: str = str(self.kg.generate(num_measures=num_measures))
        image_path: str = self.kern_2_png.render_to_image(kern, single_line=single_line)
        annotated_kern: str = convert_kern_to_annotated(kern)
        return image_path, annotated_kern


if __name__ == "__main__":
    gen = Generator()
    image_path, kern = gen.generateStaff(num_measures=6, single_line=True)
    print(kern)
