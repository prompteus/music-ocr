from pathlib import Path
import multiprocessing


def _generate_single_sample(args):
    try:
        idx, output_dir = args
        import subprocess
        import json
        import sys
        import os

        script_path = os.path.join(os.path.dirname(__file__), "worker_script.py")
        result = subprocess.run(
            [sys.executable, script_path, str(idx), output_dir],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[WARN] Sample {idx} failed. Exit code {result.returncode}.", flush=True)
            return None

        # The last line of stdout should be the JSON payload
        lines = result.stdout.strip().split("\n")
        payload = lines[-1] if lines else ""
        return json.loads(payload)

    except Exception as e:
        print(f"[WARN] Sample {args[0]} failed in parent: {e}", flush=True)
        return None


def _generate_split(split_name, total, img_dir, num_workers):

    args = [(i, str(img_dir)) for i in range(total)]
    meta = []
    failed = 0

    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for i, result in enumerate(pool.imap_unordered(_generate_single_sample, args)):
            if result is None:
                failed += 1
            else:
                meta.append(result)
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  {split_name}: {i + 1}/{total} (failed: {failed})", flush=True)

    return meta


def generate_hf_dataset(
    train_samples: int = 500,
    val_samples: int = 50,
    output_dir: str = "synthetic_omr_dataset",
    num_workers: int = 8,
) -> None:
    output_path = Path(output_dir)
    train_img_dir = output_path / "images" / "train"
    val_img_dir = output_path / "images" / "val"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating train split with {train_samples} samples...")
    train_meta = _generate_split("train", train_samples, train_img_dir, num_workers)

    print(f"Generating val split with {val_samples} samples...")
    val_meta = _generate_split("val", val_samples, val_img_dir, num_workers)

    print(f"Building dataset ({len(train_meta)} train, {len(val_meta)} val)...")
    from datasets import Dataset, DatasetDict, Features, Image as DatasetsImage, Value

    features = Features(
        {
            "image": DatasetsImage(),
            "transcription": Value("string"),
        }
    )

    train_data = {
        "image": [str(train_img_dir / m["image_file"]) for m in train_meta],
        "transcription": [m["transcription"] for m in train_meta],
    }
    val_data = {
        "image": [str(val_img_dir / m["image_file"]) for m in val_meta],
        "transcription": [m["transcription"] for m in val_meta],
    }

    train_dataset = Dataset.from_dict(train_data, features=features)
    val_dataset = Dataset.from_dict(val_data, features=features)
    dataset_dict = DatasetDict({"train": train_dataset, "val": val_dataset})

    hf_path = output_path / "hf_dataset"
    print(f"Saving HF dataset to {hf_path}...")
    dataset_dict.save_to_disk(hf_path)
    print("Done!")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="synthetic_omr_dataset")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    generate_hf_dataset(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
