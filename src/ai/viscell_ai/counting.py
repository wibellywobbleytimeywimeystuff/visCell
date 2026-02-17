"""Modul: counting.py
Beschreibung:
Dieses Skript wertet die vom KI-Modell erzeugten Heatmaps aus und zählt erkannte Zellzentren (Ery, Leuko, Hefe) anhand von Masken, Peak-Findung und Segmentierung.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der Bibliotheken
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from .config import CLASSES, TileSpec
from .peaks import find_peaks


# =========================
# 2 ) Klasse: CountParams
# =========================
# Zählparameter
@dataclass
class CountParams:
    mask_thresh: float = 0.35
    center_seed_thresh: float = 0.60

    min_peak_dist: int = 14

    min_region_area: int = 120

    class_conf_thresh: float = 0.25


# =========================
# 3 ) Crop: Maps
# =========================
# Schneidet Randbereiche aus prediction-maps heraus --> Zählung nicht verfälscht
def safe_crop_maps(centers: np.ndarray, mask: np.ndarray, tile_spec: TileSpec):
    m = int(tile_spec.safe_margin)
    if m <= 0:
        return centers, mask
    return centers[m:-m, m:-m, :], mask[m:-m, m:-m, :]


# =========================
# 4 ) Zählung
# =========================
# Zählt Zentren anhand der Heatmaps und Maske
# Maske normalisieren (0,1) → Seeds aus Heatmap → Watershed-Segmentierung → Klassenzuordnung
def count_from_maps(
    centers: np.ndarray,
    mask: np.ndarray,
    tile_spec: TileSpec,
    params: CountParams = CountParams(),
) -> Dict[str, int]:
    centers_crop, mask_crop = safe_crop_maps(centers, mask, tile_spec)

    # Maske normalisieren (0,1)
    mask_bin = (mask_crop[..., 0] >= params.mask_thresh).astype(np.uint8) * 255
    if mask_bin.max() == 0:
        return {c: 0 for c in CLASSES}

    # Seeds aus Heatmap
    center_max = np.max(centers_crop, axis=-1)
    seed_bin = (center_max >= params.center_seed_thresh).astype(np.uint8) * 255
    seed_peaks = find_peaks(seed_bin, min_dist=params.min_peak_dist)

    num_labels, seed_markers = cv2.connectedComponents(
        (seed_peaks > 0).astype(np.uint8)
    )
    if num_labels <= 1:
        return {c: 0 for c in CLASSES}

    dummy_rgb = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    markers = seed_markers.astype(np.int32)

    # Watershed-Segmentierung
    cv2.watershed(dummy_rgb, markers)

    counts = {c: 0 for c in CLASSES}
    for region_id in range(1, markers.max() + 1):
        region_mask = markers == region_id
        area = int(region_mask.sum())
        if area < params.min_region_area:
            continue

        # Klassenzuordnung
        class_scores = []
        for ci, cls in enumerate(CLASSES):
            class_scores.append(
                float(centers_crop[..., ci][region_mask].mean() if area > 0 else 0.0)
            )

        best_ci = int(np.argmax(class_scores))
        best_score = float(class_scores[best_ci])
        if best_score < params.class_conf_thresh:
            continue

        counts[CLASSES[best_ci]] += 1

    return counts
