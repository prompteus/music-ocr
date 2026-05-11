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

    BACKGROUND_COLOR: str = "#FFFFFF"
    tk: verovio.toolkit

    def __init__(self) -> None:
        verovio.enableLog(verovio.LOG_OFF)
        self.tk = verovio.toolkit()

    def __call__(self, kern_sequence: str) -> str:
        return self.render_to_image(kern_sequence)

    def render_to_image(self, kern_sequence: str, *, single_line: bool = True) -> str:
        options: dict[str, str | bool] = {
            "adjustPageHeight": True,
            "adjustPageWidth": True,
            "header": "none",
            "footer": "none",
        }
        if single_line:
            options["breaks"] = "none"
        self.tk.setOptions(options)
        self.tk.loadData(kern_sequence)
        svg_data: str = self.tk.renderToSVG(1)
        image: Image.Image = self.render(svg_data)
        import os
        import tempfile
        import uuid

        png_file_name: str = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4().hex}.png")
        image.save(png_file_name)
        return png_file_name

    def render(
        self,
        svg: str,
        width: int | None = None,
        height: int | None = None,
        resvg_cmd: str | None = None,
    ) -> Image.Image:
        if resvg_cmd is None:
            resvg_cmd = shutil.which("resvg")
            if resvg_cmd is None:
                import sys

                venv_bin: str = os.path.dirname(sys.executable)
                candidate: str = os.path.join(venv_bin, "resvg")
                if os.path.isfile(candidate):
                    resvg_cmd = candidate
            if resvg_cmd is None:
                raise RuntimeError("resvg binary not found in PATH. Install it via: uv pip install resvg-cli")

        input_bytes: bytes = svg.encode()

        with tempfile.NamedTemporaryFile(suffix=".png") as temp_png:
            command: list[str] = [
                resvg_cmd,
            ]

            if width is not None:
                command.append(f"-w={width}")

            if height is not None:
                command.append(f"-h={height}")

            command.append(f"--background={self.BACKGROUND_COLOR}")

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

            img: Image.Image = Image.open(temp_png.name)
            img.load()
            return img
