"""CLI entry point using Typer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print("typer is required. Install with: pip install typer", file=sys.stderr)
    sys.exit(1)

app = typer.Typer(name="ancient-ocr", help="Arabic OCR + Lexicon-Augmented Intelligence")

_MODES = ["clean", "annotated", "debug"]


@app.command("process")
def process(
    file_path: Path = typer.Argument(..., help="Input PDF or image file"),
    mode: str = typer.Option("clean", "--mode", "-m", help="Output mode: clean|annotated|debug"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write output to file"),
    profile: str = typer.Option("default", "--profile", "-p", help="OCR profile name"),
):
    """Process a single file through the OCR pipeline."""
    if mode not in _MODES:
        typer.echo(f"Error: invalid mode '{mode}'. Must be one of: {_MODES}", err=True)
        raise typer.Exit(1)
    from cli.commands import handle_process
    code = handle_process(str(file_path), mode, str(output) if output else None,
                          profile_name=profile)
    raise typer.Exit(code)


@app.command("batch")
def batch(
    folder: Path = typer.Argument(..., help="Folder containing input files"),
    mode: str = typer.Option("clean", "--mode", "-m"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
):
    """Process all supported files in a folder."""
    from cli.commands import handle_batch
    code = handle_batch(str(folder), mode, str(output_dir) if output_dir else None)
    raise typer.Exit(code)


@app.command("debug")
def debug_cmd(
    file_path: Path = typer.Argument(..., help="Input file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
):
    """Process a file in debug mode (full pipeline trace)."""
    from cli.commands import handle_process
    code = handle_process(str(file_path), "debug", str(output) if output else None)
    raise typer.Exit(code)


@app.command("feedback")
def feedback(
    image_path: str = typer.Argument(..., help="Path to the token crop image"),
    predicted: str = typer.Option(..., "--predicted", "-p"),
    ground_truth: str = typer.Option(..., "--ground-truth", "-g"),
    source_file: str = typer.Option(..., "--source", "-s"),
    bbox: str = typer.Option("0,0,0,0", "--bbox", "-b", help="x,y,w,h in page space"),
    page_index: int = typer.Option(0, "--page"),
):
    """Submit a ground-truth correction for a predicted token."""
    try:
        bx = tuple(int(v) for v in bbox.split(","))
        assert len(bx) == 4
    except Exception:
        typer.echo("Error: bbox must be 'x,y,w,h'", err=True)
        raise typer.Exit(1)

    from cli.commands import handle_feedback
    code = handle_feedback(image_path, bx, page_index, predicted, ground_truth, source_file)
    raise typer.Exit(code)


@app.command("upload-lexicons")
def upload_lexicons(
    repo_id: str = typer.Option("", "--repo-id", "-r", help="HuggingFace dataset repo, e.g. username/ancient-ocr-lexicons"),
    config_path: str = typer.Option("config.yaml", "--config"),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip rebuild; upload existing lexicons.db"),
):
    """Build lexicons.db from all enabled sources, then upload to HuggingFace."""
    from cli.commands import handle_upload_lexicons
    code = handle_upload_lexicons(repo_id=repo_id, config_path=config_path, skip_build=skip_build)
    raise typer.Exit(code)


@app.command("calibrate")
def calibrate(
    apply: bool = typer.Option(False, "--apply", help="Write suggested weights to config.yaml"),
    config_path: str = typer.Option("config.yaml", "--config"),
):
    """Calibrate scoring weights from stored feedback corrections."""
    from cli.commands import handle_calibrate
    code = handle_calibrate(apply=apply, config_path=config_path)
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
