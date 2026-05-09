from pathlib import Path


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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    args = parser.parse_args()

    upload_to_hf(dataset_path=args.dataset_path, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
