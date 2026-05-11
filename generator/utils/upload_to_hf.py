from pathlib import Path
import typer


def upload_to_hf(dataset_path: str, repo_id: str) -> None:
    from datasets import load_from_disk

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    print(f"Loading dataset from {path}...")
    dataset = load_from_disk(str(path))

    print(f"Pushing to Hugging Face Hub as '{repo_id}'...")
    dataset.push_to_hub(repo_id)
    print("Upload complete!")


def main(
    dataset_path: str = typer.Option(..., help="Path to the HF dataset on disk."),
    repo_id: str = typer.Option(..., help="Hugging Face repository ID to push to."),
) -> None:
    upload_to_hf(dataset_path=dataset_path, repo_id=repo_id)


if __name__ == "__main__":
    typer.run(main)
