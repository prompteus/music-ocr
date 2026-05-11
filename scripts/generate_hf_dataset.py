import multiprocessing
import typer
from music_ocr.generator.dataset import generate_hf_dataset


def main(
    train_samples: int = typer.Option(500, help="Number of training samples to generate."),
    val_samples: int = typer.Option(50, help="Number of validation samples to generate."),
    output_dir: str = typer.Option("synthetic_omr_dataset", help="Output directory for the dataset."),
    num_workers: int = typer.Option(8, help="Number of parallel workers."),
) -> None:
    generate_hf_dataset(
        train_samples=train_samples,
        val_samples=val_samples,
        output_dir=output_dir,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    typer.run(main)
