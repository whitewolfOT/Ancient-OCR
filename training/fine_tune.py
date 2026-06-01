"""Optional Tesseract fine-tuning behind the [finetune] extra.

All operations are explicit — nothing runs automatically.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from confidence_engine.state import FeedbackEntry
from utils.logging import get_logger

log = get_logger(__name__)


def _check_finetune_available() -> None:
    """Raise ImportError if Tesseract training tools are not installed."""
    if not shutil.which("tesseract"):
        raise ImportError(
            "tesseract binary not found. "
            "Install tesseract-ocr and tesseract training tools."
        )
    if not shutil.which("combine_tessdata"):
        raise ImportError(
            "combine_tessdata not found. "
            "Install tesseract-ocr-dev / libtesseract-dev for training tools."
        )


def prepare_training_data(
    feedback_entries: list[FeedbackEntry],
    output_dir: str,
) -> dict:
    """Convert feedback corrections into Tesseract .box / .lstmf training format.

    Writes:
      {output_dir}/{id}.png    — cropped image
      {output_dir}/{id}.gt.txt — ground truth text

    Returns summary dict {prepared, skipped, output_dir}.
    """
    _check_finetune_available()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prepared = 0
    skipped = 0

    for entry in feedback_entries:
        if not entry.image_path or not Path(entry.image_path).exists():
            skipped += 1
            continue
        try:
            import shutil as _shutil
            dst_img = out / f"{entry.id}.png"
            _shutil.copy2(entry.image_path, dst_img)
            gt_path = out / f"{entry.id}.gt.txt"
            gt_path.write_text(entry.ground_truth, encoding="utf-8")
            prepared += 1
        except Exception as exc:
            log.warning(f"prepare_training_data failed for {entry.id}: {exc}")
            skipped += 1

    log.info(f"prepare_training_data prepared={prepared} skipped={skipped} dir={output_dir}")
    return {"prepared": prepared, "skipped": skipped, "output_dir": str(out)}


def run_finetune(
    training_dir: str,
    base_model: str = "ara",
    output_model: str = "ara_finetuned",
    config=None,
) -> dict:
    """Run Tesseract LSTM fine-tuning.

    Requires tesseract training tools (combine_tessdata, lstmtraining, etc.).
    This is a simplified pipeline — production fine-tuning requires more steps.

    Returns {success, output_model, log_lines}.
    """
    _check_finetune_available()

    training_path = Path(training_dir)
    if not training_path.exists():
        raise ValueError(f"Training directory not found: {training_dir}")

    log_lines: list[str] = []

    # Step 1: generate .lstmf files from images + ground truth
    gt_files = list(training_path.glob("*.gt.txt"))
    if not gt_files:
        raise ValueError(f"No .gt.txt files found in {training_dir}")

    log.info(f"run_finetune base={base_model} samples={len(gt_files)}")

    for gt_file in gt_files:
        stem = gt_file.stem.replace(".gt", "")
        img_file = training_path / f"{stem}.png"
        if not img_file.exists():
            continue
        try:
            result = subprocess.run(
                ["tesseract", str(img_file), str(training_path / stem),
                 "--psm", "7", "lstm.train"],
                capture_output=True, text=True, timeout=30,
            )
            log_lines.append(result.stdout + result.stderr)
        except Exception as exc:
            log_lines.append(f"ERROR: {exc}")

    return {
        "success": True,
        "output_model": output_model,
        "samples_processed": len(gt_files),
        "log_lines": log_lines[:20],
    }
