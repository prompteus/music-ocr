from typing import Tuple, Optional

from generator.utils.kern_to_png import KernToPng
from generator.utils.kern_generator import KernGenerator
from generator.utils.kern_annotation_parser import convert_kern_to_annotated


class Generator:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.kg = KernGenerator() if seed is None else KernGenerator(seed)
        self.kern_2_png = KernToPng()

    def generateStaff(self) -> Tuple[str, str]:
        kern: str = str(self.kg.generate(num_measures=4))
        image_path: str = self.kern_2_png.render_to_image(kern)
        annotated_kern: str = convert_kern_to_annotated(kern)
        return image_path, annotated_kern


if __name__ == "__main__":
    gen = Generator()
    image_path, kern = gen.generateStaff()
    print(kern)
