"""
Profile loader — reads config/profiles.yaml and provides typed OCRProfile objects.
No import-time side effects. No model loading here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PreprocessingParams:
    brightness: int = 0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    stroke_normalization_enabled: bool = False
    stroke_target_width: int = 2
    denoise_strength: int = 0
    sharpen: float = 0.0


@dataclass
class OCRProfile:
    name: str
    description: str = ""
    binarizer: str = "otsu"          # "otsu" | "nlbin" | "sauvola"
    seg_model: str = "zenodo:14295555"
    rec_model: str = "agapet"
    rec_model_secondary: str | None = None
    n_best: int = 3
    rtl: bool = True
    device: str = "cpu"
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)


class ProfileManager:
    """Load, save, and serve OCRProfile objects from a YAML file."""

    def __init__(self, profiles_path: Path):
        self._path = profiles_path
        self._profiles: dict[str, OCRProfile] = {}
        self.load()

    def load(self) -> None:
        """Re-read the YAML file. Safe to call at any time (hot-reload)."""
        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._profiles = {}
        for name, cfg in data.get("profiles", {}).items():
            pp = cfg.get("preprocessing", {})
            sn = pp.get("stroke_normalization", {})
            pre = PreprocessingParams(
                brightness=int(pp.get("brightness", 0)),
                contrast=float(pp.get("contrast", 1.0)),
                gamma=float(pp.get("gamma", 1.0)),
                saturation=float(pp.get("saturation", 1.0)),
                stroke_normalization_enabled=bool(sn.get("enabled", False)),
                stroke_target_width=int(sn.get("target_width", 2)),
                denoise_strength=int(pp.get("denoise_strength", 0)),
                sharpen=float(pp.get("sharpen", 0.0)),
            )
            self._profiles[name] = OCRProfile(
                name=name,
                description=cfg.get("description", ""),
                binarizer=cfg.get("binarizer", "otsu"),
                seg_model=cfg.get("seg_model", "zenodo:14295555"),
                rec_model=cfg.get("rec_model", "agapet"),
                rec_model_secondary=cfg.get("rec_model_secondary"),
                n_best=int(cfg.get("n_best", 3)),
                rtl=bool(cfg.get("rtl", True)),
                device=cfg.get("device", "cpu"),
                preprocessing=pre,
            )

    def save(self) -> None:
        """Write current profiles back to YAML."""
        data: dict = {"profiles": {}}
        for name, p in self._profiles.items():
            pr = p.preprocessing
            data["profiles"][name] = {
                "description": p.description,
                "binarizer": p.binarizer,
                "seg_model": p.seg_model,
                "rec_model": p.rec_model,
                "rec_model_secondary": p.rec_model_secondary,
                "n_best": p.n_best,
                "rtl": p.rtl,
                "device": p.device,
                "preprocessing": {
                    "brightness": pr.brightness,
                    "contrast": pr.contrast,
                    "gamma": pr.gamma,
                    "saturation": pr.saturation,
                    "stroke_normalization": {
                        "enabled": pr.stroke_normalization_enabled,
                        "target_width": pr.stroke_target_width,
                    },
                    "denoise_strength": pr.denoise_strength,
                    "sharpen": pr.sharpen,
                },
            }
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def get(self, name: str) -> OCRProfile:
        """Return named profile, or 'default' if not found."""
        return copy.deepcopy(self._profiles.get(name, self._profiles["default"]))

    def list(self) -> list[str]:
        return list(self._profiles.keys())

    def upsert(self, profile: OCRProfile) -> None:
        """Add or update a profile in memory. Call save() to persist."""
        self._profiles[profile.name] = copy.deepcopy(profile)

    def delete(self, name: str) -> bool:
        """Remove a profile. Returns False if name is 'default' (protected)."""
        if name == "default":
            return False
        removed = self._profiles.pop(name, None)
        return removed is not None
