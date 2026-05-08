import os
import shutil
import subprocess
import tempfile

import verovio
from PIL import Image


class KernToPng:
    """
    Renders Humdrum **kern (or PRAIG annotated ekern) sequences to PNG images.
    Uses Verovio to compile the notation and PyMuPDF (fitz) to rasterize it flawlessly.
    """

    def __init__(self):
        verovio.enableLog(verovio.LOG_OFF)
        self.tk = verovio.toolkit()

    def __call__(self, kern_sequence: str) -> str:
        return self.render_to_image(kern_sequence)

    def render_to_image(self, kern_sequence: str) -> str:
        self.tk.loadData(kern_sequence)
        self.tk.setOptions({"adjustPageHeight": True, "adjustPageWidth": True, "header": "none", "footer": "none"})
        svg_data = self.tk.renderToSVG(1)
        image = self.render(svg_data)
        png_file_name = "output.png"
        image.save(png_file_name)
        return png_file_name

    def render(
        self,
        svg: str,
        width: int | None = None,
        height: int | None = None,
        background: str | None = None,
        resvg_cmd: str | None = None,
    ) -> Image.Image:
        """
        Render an SVG to a PIL image using resvg.

        If background is None, the image will have an alpha channel.
        If both width and height are None, the image will be rendered at the SVG's specified size.
        """
        if resvg_cmd is None:
            resvg_cmd = shutil.which("resvg")
            if resvg_cmd is None:
                import sys

                venv_bin = os.path.dirname(sys.executable)
                candidate = os.path.join(venv_bin, "resvg")
                if os.path.isfile(candidate):
                    resvg_cmd = candidate
            if resvg_cmd is None:
                raise RuntimeError("resvg binary not found in PATH. Install it via: uv pip install resvg-cli")

        input_bytes = svg.encode()

        with tempfile.NamedTemporaryFile(suffix=".png") as temp_png:
            command = [
                resvg_cmd,
            ]

            if width is not None:
                command.append(f"-w={width}")

            if height is not None:
                command.append(f"-h={height}")

            if background is not None:
                command.append(f"--background={background}")

            # Read SVG from stdin
            command.append("-")
            command.append(temp_png.name)

            subprocess.run(
                command,
                input=input_bytes,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )

            return Image.open(temp_png.name)


if __name__ == "__main__":
    print("TODO tests")
