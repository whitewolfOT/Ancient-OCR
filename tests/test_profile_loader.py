"""Tests for ocr_engine/profile_loader.py."""
import copy
from pathlib import Path

import pytest
import yaml

from ocr_engine.profile_loader import OCRProfile, PreprocessingParams, ProfileManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_YAML = """
profiles:
  default:
    description: "Test default"
    binarizer: "otsu"
    seg_model: "zenodo:14295555"
    rec_model: "agapet"
    rec_model_secondary: null
    n_best: 3
    rtl: true
    device: "cpu"
    preprocessing:
      brightness: 0
      contrast: 1.0
      gamma: 1.0
      saturation: 1.0
      stroke_normalization:
        enabled: false
        target_width: 2
      denoise_strength: 0
      sharpen: 0.0

  custom_a:
    description: "Custom profile A"
    binarizer: "sauvola"
    seg_model: "zenodo:14295555"
    rec_model: "muharaf"
    rec_model_secondary: "agapet"
    n_best: 5
    rtl: true
    device: "cpu"
    preprocessing:
      brightness: 10
      contrast: 1.2
      gamma: 0.9
      saturation: 1.0
      stroke_normalization:
        enabled: true
        target_width: 3
      denoise_strength: 12
      sharpen: 0.4

  custom_b:
    description: "Custom profile B"
    binarizer: "nlbin"
    seg_model: "zenodo:14295555"
    rec_model: "agapet"
    rec_model_secondary: null
    n_best: 2
    rtl: false
    device: "cpu"
    preprocessing:
      brightness: -5
      contrast: 0.9
      gamma: 1.1
      saturation: 0.5
      stroke_normalization:
        enabled: false
        target_width: 2
      denoise_strength: 3
      sharpen: 0.1
"""


@pytest.fixture
def tmp_profiles_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "profiles.yaml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_profiles(tmp_profiles_yaml):
    mgr = ProfileManager(tmp_profiles_yaml)
    assert len(mgr.list()) == 3
    assert "default" in mgr.list()
    assert "custom_a" in mgr.list()
    assert "custom_b" in mgr.list()


def test_get_returns_default_for_unknown(tmp_profiles_yaml):
    mgr = ProfileManager(tmp_profiles_yaml)
    p = mgr.get("nonexistent_profile_xyz")
    assert p.name == "default"


def test_upsert_and_save(tmp_profiles_yaml):
    mgr = ProfileManager(tmp_profiles_yaml)
    new_profile = OCRProfile(
        name="saved_test",
        description="A saved profile",
        binarizer="sauvola",
        n_best=7,
    )
    mgr.upsert(new_profile)
    mgr.save()

    # Reload from disk and verify
    mgr2 = ProfileManager(tmp_profiles_yaml)
    assert "saved_test" in mgr2.list()
    reloaded = mgr2.get("saved_test")
    assert reloaded.n_best == 7
    assert reloaded.binarizer == "sauvola"
    assert reloaded.description == "A saved profile"


def test_delete_default_forbidden(tmp_profiles_yaml):
    mgr = ProfileManager(tmp_profiles_yaml)
    result = mgr.delete("default")
    assert result is False
    assert "default" in mgr.list()


def test_preprocessing_params_typed(tmp_profiles_yaml):
    mgr = ProfileManager(tmp_profiles_yaml)
    p = mgr.get("custom_a")
    assert isinstance(p.preprocessing.brightness, int)
    assert isinstance(p.preprocessing.gamma, float)
    assert isinstance(p.preprocessing.stroke_normalization_enabled, bool)
    assert isinstance(p.preprocessing.denoise_strength, int)
    assert isinstance(p.preprocessing.sharpen, float)
    assert p.preprocessing.brightness == 10
    assert p.preprocessing.gamma == 0.9
    assert p.preprocessing.stroke_normalization_enabled is True
    assert p.preprocessing.stroke_target_width == 3


def test_get_returns_deep_copy(tmp_profiles_yaml):
    mgr = ProfileManager(tmp_profiles_yaml)
    p1 = mgr.get("default")
    p1.preprocessing.brightness = 999
    p2 = mgr.get("default")
    assert p2.preprocessing.brightness == 0  # original unaffected
