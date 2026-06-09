"""CLI command handlers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from utils.logging import get_logger

log = get_logger(__name__)


def handle_process(file_path: str, mode: str, output: str | None,
                   profile_name: str = "default") -> int:
    """Process a single file. Returns exit code."""
    try:
        from main import process_file
        result = process_file(file_path, mode, profile_name=profile_name)
        _write_or_print(result, output, mode)
        return 0
    except Exception as exc:
        _error(str(exc), file_path)
        return 1


def handle_batch(folder: str, mode: str, output_dir: str | None) -> int:
    """Process all supported files in a folder."""
    from ingest.document_loader import SUPPORTED_EXTENSIONS
    folder_path = Path(folder)
    if not folder_path.is_dir():
        _error(f"Not a directory: {folder}")
        return 1

    files = [f for f in folder_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        print(f"No supported files found in {folder}", file=sys.stderr)
        return 0

    success, failed = 0, 0
    for f in sorted(files):
        out = str(Path(output_dir) / f"{f.stem}.json") if output_dir else None
        if out:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        code = handle_process(str(f), mode, out)
        if code == 0:
            success += 1
        else:
            failed += 1

    print(f"Batch complete: {success} ok, {failed} failed", file=sys.stderr)
    return 0 if failed == 0 else 1


def handle_feedback(
    image_path: str,
    bbox: tuple[int, int, int, int],
    page_index: int,
    predicted: str,
    ground_truth: str,
    source_file: str,
) -> int:
    """Submit a feedback correction."""
    try:
        from confidence_engine.state import FeedbackEntry
        from training.feedback_store import submit
        from datetime import datetime, timezone

        entry = FeedbackEntry(
            id="",
            image_path=image_path,
            bbox=bbox,
            page_index=page_index,
            predicted=predicted,
            ground_truth=ground_truth,
            source_file=source_file,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        entry_id = submit(entry)
        print(json.dumps({"status": "ok", "id": entry_id}, ensure_ascii=False))
        return 0
    except Exception as exc:
        _error(str(exc))
        return 1


def handle_upload_lexicons(
    repo_id: str,
    config_path: str = "config.yaml",
    skip_build: bool = False,
) -> int:
    """Build lexicons.db and upload to HuggingFace. Returns exit code."""
    try:
        from utils.config import get_config
        from lexicon_ingestion.downloader import (
            build_lexicons_db,
            upload_to_hf,
            _db_path,
            _hf_repo,
            _hf_token_env_name,
            _count_entries,
            _MIN_ENTRIES_FOR_UPLOAD,
        )
        import os

        config = get_config()
        effective_repo_id = repo_id or _hf_repo(config)
        if not effective_repo_id:
            print(
                "Error: no repo_id provided. Pass --repo-id or set "
                "lexicon.hf_repo_id in config.yaml",
                file=sys.stderr,
            )
            return 1

        token_env = _hf_token_env_name(config)
        token = os.environ.get(token_env)
        if not token:
            print(
                f"Error: HuggingFace token not found. "
                f"Set the {token_env} environment variable.",
                file=sys.stderr,
            )
            return 1

        db = _db_path(config)

        if not skip_build:
            print("Building lexicons.db from all enabled sources…", file=sys.stderr)
            try:
                total = build_lexicons_db(output_path=db, config=config)
                print(f"Build complete: {total} entries written to {db}", file=sys.stderr)
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1

        count, sources = _count_entries(db)
        if count < _MIN_ENTRIES_FOR_UPLOAD:
            print(
                f"Error: DB has only {count} entries (minimum {_MIN_ENTRIES_FOR_UPLOAD}). "
                "Aborting upload.",
                file=sys.stderr,
            )
            return 1

        print(f"Uploading {db} ({count} entries from {sources}) → {effective_repo_id}…",
              file=sys.stderr)
        upload_to_hf(db, effective_repo_id, token=token, config=config)
        print("Upload complete.", file=sys.stderr)
        return 0

    except Exception as exc:
        _error(str(exc))
        return 1


def handle_calibrate(apply: bool = False, config_path: str = "config.yaml") -> int:
    """Run calibration and optionally apply suggested weights."""
    try:
        from training.calibrator import calibrate, apply_weights
        from utils.config import get_config

        result = calibrate(get_config())
        out = {
            "sample_size": result.sample_size,
            "current_weights": result.current_weights,
            "suggested_weights": result.suggested_weights,
            "delta": result.delta,
        }
        if result.warning:
            out["warning"] = result.warning
        print(json.dumps(out, indent=2, ensure_ascii=False))

        if apply and not result.warning:
            apply_weights(result, config_path)
            print(f"Weights written to {config_path}", file=sys.stderr)
        elif apply and result.warning:
            print(f"Cannot apply: {result.warning}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        _error(str(exc))
        return 1


def _write_or_print(result: dict, output: str | None, mode: str):
    if mode == "clean" and not output:
        print(result.get("text", ""))
        return
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(payload, encoding="utf-8")
        print(f"Written to {output}", file=sys.stderr)
    else:
        print(payload)


def _error(msg: str, file: str | None = None):
    err = {"error": msg}
    if file:
        err["file"] = file
    print(json.dumps(err, ensure_ascii=False), file=sys.stderr)
