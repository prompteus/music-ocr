import typer
import json
import random
import os


def main(
    idx: int = typer.Argument(..., help="The index of the sample being generated."),
    output_dir: str = typer.Argument(..., help="The directory where the generated image will be saved."),
):
    from music_ocr.generator.generator_api import Generator
    from PIL import Image

    gen = Generator()
    num_measures = random.randint(2, 6)
    image_path, annotated_kern = gen.generateStaff(num_measures=num_measures, single_line=True)

    img = Image.open(image_path)
    img.load()

    dest_path = os.path.join(output_dir, f"{idx:07d}.png")
    img.save(dest_path)

    try:
        os.remove(image_path)
    except OSError:
        pass

    result = {"idx": idx, "image_file": f"{idx:07d}.png", "transcription": annotated_kern}
    print(json.dumps(result))


if __name__ == "__main__":
    typer.run(main)
