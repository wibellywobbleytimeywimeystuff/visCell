"""
Modul: peaks.py
Beschreibung:
Dieses Skript verarbeitet Mikroskopiebilder und unterstützt die Extraktion relevanter Bildinformationen.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der Bildverarbeitung
# =========================

from __future__ import annotations




import cv2
import numpy as np



# =========================
# 2 ) Find Peaks
# =========================

def find_peaks(binary_map_u8: np.ndarray, min_dist: int = 12) -> np.ndarray:
    
    if binary_map_u8.dtype != np.uint8:
        binary_map_u8 = binary_map_u8.astype(np.uint8)

    kernel_3 = np.ones((3, 3), np.uint8)
    binary_clean = cv2.morphologyEx(binary_map_u8, cv2.MORPH_OPEN, kernel_3, iterations=1)

    dist_map = cv2.distanceTransform((binary_clean > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist_map.max()) <= 0.0:
        return np.zeros_like(binary_map_u8, dtype=np.uint8)

    k = max(3, int(min_dist) | 1)
    dilated = cv2.dilate(dist_map, np.ones((k, k), np.uint8))
    peak_mask = (dist_map == dilated) & (dist_map > 0)

    peak_u8 = (peak_mask.astype(np.uint8) * 255)
    peak_u8[binary_clean == 0] = 0
    return peak_u8