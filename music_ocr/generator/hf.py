from pathlib import Path


def upload_to_hf(dataset_path: str | Path, repo_id: str) -> None:
    from datasets import load_from_disk

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    print(f"Loading dataset from {path}...")
    dataset = load_from_disk(str(path))

    print(f"Pushing to Hugging Face Hub as '{repo_id}'...")
    dataset.push_to_hub(repo_id)
    print("Upload complete!")
