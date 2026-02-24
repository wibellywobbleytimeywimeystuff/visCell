"""Modul: peaks.py
Beschreibung:
Robuste Peak-Erkennung ohne Distance-Transform/Wateshed.
Erkennt lokale Maxima in Float-Heatmaps per OpenCV-Dilatation.

Autor: Marlon Aust
Projektname: visCell
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PeakParams:
    min_peak_dist: int = 18           # Pixel (ungefähr Zellradius)
    max_component_area: int = 25      # Pixel (Plateau-Schutz)
    abs_seed_thresh: float = 0.03     # absolute Mindestschwelle
    quantile: float = 0.995           # Perzentil-Schwelle (Top 0.5%)
    max_fg_frac: float = 0.01         # max. Anteil Pixel >= thr


def robust_threshold(hm: np.ndarray, params: PeakParams) -> float:
    hm_f = hm.astype(np.float32)
    mx = float(hm_f.max()) if hm_f.size else 0.0
    if mx <= 1e-8:
        return 1.0

    # Start: quantile, dann ggf. hochziehen, wenn zu viel Fläche über thr liegt (Plateau)
    for q in (params.quantile, 0.997, 0.999, 0.9995):
        thr = float(np.quantile(hm_f, q))
        thr = max(params.abs_seed_thresh, thr)
        fg = float((hm_f >= thr).mean())
        if fg <= params.max_fg_frac:
            return thr
    thr = float(np.quantile(hm_f, params.quantile))
    return max(params.abs_seed_thresh, thr)


def peak_mask(hm: np.ndarray, params: PeakParams, thr: float) -> np.ndarray:
    k = max(3, int(params.min_peak_dist) * 2 + 1)
    kernel = np.ones((k, k), np.uint8)
    hm_f = hm.astype(np.float32)
    dil = cv2.dilate(hm_f, kernel)
    return (hm_f >= thr) & (hm_f == dil)


def count_peaks(hm: np.ndarray, params: PeakParams) -> int:
    if hm.size == 0 or float(hm.max()) <= 1e-8:
        return 0
    thr = robust_threshold(hm, params)
    peaks = peak_mask(hm, params, thr)

    # Wenn extrem viel Peak-Fläche, ist es kein sinnvolles Peak-Signal
    if float(peaks.mean()) > 0.02:
        return 0

    peaks_u8 = (peaks.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((peaks_u8 > 0).astype(np.uint8), connectivity=8)
    cnt = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= params.max_component_area:
            cnt += 1
    return cnt
