import typer
from music_ocr.generator.hf import upload_to_hf


def main(
    dataset_path: str = typer.Option(..., help="Path to the HF dataset on disk."),
    repo_id: str = typer.Option(..., help="Hugging Face repository ID to push to."),
) -> None:
    upload_to_hf(dataset_path=dataset_path, repo_id=repo_id)


if __name__ == "__main__":
    typer.run(main)
