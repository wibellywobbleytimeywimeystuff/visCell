"""Modul: counting.py
Beschreibung:
Robustes Zählen aus Centers-Heatmaps ohne Mask-Gating, ohne Watershed.
Diese Version ist bewusst stabil gegen flache/plateauartige Heatmaps und verhindert 0/0/0
durch fragile Binär-Morphologie.

Autor: Marlon Aust
Projektname: visCell
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import cv2

from .config import CLASSES, TileSpec
from .peaks import PeakParams, count_peaks


@dataclass
class CountParams:
    # Peak-Parameter (werden an peaks.py durchgereicht)
    min_peak_dist: int = 18
    max_component_area: int = 25
    abs_seed_thresh: float = 0.03
    quantile: float = 0.995
    max_fg_frac: float = 0.01


# =========================
# 3 ) Crop: Maps
# =========================
def safe_crop_maps(centers: np.ndarray, mask: np.ndarray, tile_spec: TileSpec):
    m = int(getattr(tile_spec, "safe_margin", 0) or 0)
    if m <= 0:
        return centers, mask
    return centers[m:-m, m:-m, :], mask[m:-m, m:-m, :]


# =========================
# 4 ) Zählung (robust)
# =========================
def count_from_maps(
    centers: np.ndarray,
    mask: np.ndarray,
    tile_spec: TileSpec,
    params: CountParams = CountParams(),
) -> Dict[str, int]:
    centers_crop, _mask_crop = safe_crop_maps(centers, mask, tile_spec)

    # Defensive: falls Modell nur 1 Kanal liefert
    if centers_crop.ndim != 3 or centers_crop.shape[-1] < len(CLASSES):
        raise ValueError(f"centers shape expected (H,W,{len(CLASSES)}), got {centers_crop.shape}")

    peak_params = PeakParams(
        min_peak_dist=int(params.min_peak_dist),
        max_component_area=int(params.max_component_area),
        abs_seed_thresh=float(params.abs_seed_thresh),
        quantile=float(params.quantile),
        max_fg_frac=float(params.max_fg_frac),
    )

    counts = {c: 0 for c in CLASSES}
    for ci, cls in enumerate(CLASSES):
        hm = centers_crop[..., ci].astype(np.float32)
        counts[cls] = int(count_peaks(hm, peak_params))
    return counts
