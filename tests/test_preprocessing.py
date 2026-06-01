"""Tests for preprocessing.image_pipeline using synthetic numpy images."""
import numpy as np
import pytest

from preprocessing.image_pipeline import preprocess_image


def _make_image(h=100, w=80):
    """White BGR image."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_returns_same_shape():
    img = _make_image(100, 80)
    out, meta = preprocess_image(img)
    assert out.shape[:2] == img.shape[:2]


def test_returns_metadata_keys():
    img = _make_image()
    _, meta = preprocess_image(img)
    assert "denoise" in meta
    assert "clahe" in meta
    assert "deskew" in meta
    assert "binarize" in meta


def test_metadata_values_valid():
    img = _make_image()
    _, meta = preprocess_image(img)
    valid = {"applied", "skipped", "failed"}
    for k, v in meta.items():
        assert v in valid, f"step '{k}' has unexpected status '{v}'"


def test_output_is_numpy():
    img = _make_image()
    out, _ = preprocess_image(img)
    assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# Steps applied by default
# ---------------------------------------------------------------------------

def test_binarize_always_applied():
    """binarize is hardcoded enabled=True, so it must be applied or failed, never skipped."""
    img = _make_image()
    _, meta = preprocess_image(img)
    assert meta["binarize"] in ("applied", "failed")


# ---------------------------------------------------------------------------
# Disabling a step via config mock
# ---------------------------------------------------------------------------

class _StepCfg:
    def __init__(self, enabled):
        self.enabled = enabled


class _PreCfg:
    def __init__(self, *, denoise=True, clahe=True, deskew=True):
        self.denoise = _StepCfg(denoise)
        self.clahe = _StepCfg(clahe)
        self.deskew = _StepCfg(deskew)


class _MockConfig:
    def __init__(self, *, denoise=True, clahe=True, deskew=True):
        self.preprocessing = _PreCfg(denoise=denoise, clahe=clahe, deskew=deskew)


def test_disable_denoise_sets_skipped():
    img = _make_image()
    cfg = _MockConfig(denoise=False)
    _, meta = preprocess_image(img, config=cfg)
    assert meta["denoise"] == "skipped"


def test_disable_clahe_sets_skipped():
    img = _make_image()
    cfg = _MockConfig(clahe=False)
    _, meta = preprocess_image(img, config=cfg)
    assert meta["clahe"] == "skipped"


def test_disable_deskew_sets_skipped():
    img = _make_image()
    cfg = _MockConfig(deskew=False)
    _, meta = preprocess_image(img, config=cfg)
    assert meta["deskew"] == "skipped"


def test_disable_all_optional_steps():
    img = _make_image()
    cfg = _MockConfig(denoise=False, clahe=False, deskew=False)
    _, meta = preprocess_image(img, config=cfg)
    assert meta["denoise"] == "skipped"
    assert meta["clahe"] == "skipped"
    assert meta["deskew"] == "skipped"
    # binarize is always run
    assert meta["binarize"] in ("applied", "failed")


# ---------------------------------------------------------------------------
# Failed step does not propagate exception
# ---------------------------------------------------------------------------

def test_failed_step_does_not_raise():
    """Inject a bad image to force a step failure; pipeline must not raise."""
    # A 1D array will cause CV ops to fail
    bad_img = np.array([1, 2, 3], dtype=np.uint8)
    try:
        out, meta = preprocess_image(bad_img)
        # If it runs, at least one step should be failed or the output is returned
        for v in meta.values():
            assert v in ("applied", "skipped", "failed")
    except Exception:
        # Totally unexpected raise — test should not reach here
        pytest.fail("preprocess_image raised an exception on bad input")


# ---------------------------------------------------------------------------
# None config uses defaults (all steps enabled)
# ---------------------------------------------------------------------------

def test_none_config_uses_defaults():
    img = _make_image()
    _, meta = preprocess_image(img, config=None)
    # With defaults all optional steps should be applied or failed (never skipped)
    for step in ("denoise", "clahe", "deskew"):
        assert meta[step] in ("applied", "failed"), \
            f"step '{step}' was unexpectedly skipped with no config"
