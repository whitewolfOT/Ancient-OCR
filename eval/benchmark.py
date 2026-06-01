"""Benchmark runner: finds paired PNG+ref.txt files and measures CER/WER."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def run_benchmark(data_dir: str, config=None) -> dict:
    """Run the OCR pipeline on every paired {name}.png + {name}.ref.txt in data_dir.

    Discovery:
        Walks data_dir for *.png files that have a corresponding *.ref.txt sibling.

    For each pair:
        1. Run main.process_file(png_path, mode="clean") to get predicted text.
        2. Read reference text from the .ref.txt file.
        3. Compute CER and WER vs the reference.

    Returns:
        {
            "files_processed": int,
            "mean_cer": float,
            "mean_wer": float,
            "results": [
                {
                    "file": str,
                    "cer": float,
                    "wer": float,
                    "predicted": str,
                    "reference": str,
                    "error": str | None,   # set if processing failed
                }
            ]
        }

    Gracefully handles individual file failures: records them with error set
    and excludes them from mean_cer/mean_wer.
    """
    from eval.metrics import cer as compute_cer, wer as compute_wer

    data_path = Path(data_dir)
    results: list[dict[str, Any]] = []

    # Discover paired files
    png_files = sorted(data_path.glob("*.png"))
    pairs: list[tuple[Path, Path]] = []
    for png in png_files:
        ref = png.with_suffix(".ref.txt")
        if ref.exists():
            pairs.append((png, ref))

    for png_path, ref_path in pairs:
        entry: dict[str, Any] = {
            "file": str(png_path),
            "cer": None,
            "wer": None,
            "predicted": None,
            "reference": None,
            "error": None,
        }
        try:
            reference = ref_path.read_text(encoding="utf-8").strip()
            entry["reference"] = reference

            from main import process_file
            output = process_file(str(png_path), mode="clean")
            predicted = output.get("text", "")
            entry["predicted"] = predicted

            entry["cer"] = compute_cer(reference, predicted)
            entry["wer"] = compute_wer(reference, predicted)
        except Exception as exc:
            entry["error"] = str(exc)

        results.append(entry)

    # Compute means over successful results only
    successful = [r for r in results if r["error"] is None and r["cer"] is not None]
    if successful:
        mean_cer = sum(r["cer"] for r in successful) / len(successful)
        mean_wer = sum(r["wer"] for r in successful) / len(successful)
    else:
        mean_cer = 0.0
        mean_wer = 0.0

    return {
        "files_processed": len(successful),
        "mean_cer": round(mean_cer, 4),
        "mean_wer": round(mean_wer, 4),
        "results": results,
    }
