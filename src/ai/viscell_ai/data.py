"""Modul: data.py
Beschreibung:
Dieses Skript lädt Punkt-Labels, liest Bilder ein, zerlegt Bilder in Tiles und erzeugt daraus Trainingsdaten sowie Ziel-Heatmaps und Masken.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der Bibliotheken
# =========================
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import CLASSES, TileSpec


# =========================
# 2 ) Import und Laden: JSONs
# =========================
def load_points_json(json_path: Path) -> Tuple[str, List[dict]]:
    """
    Lädt eine *_points.json und normalisiert die Klassenbezeichnung.

    In euren JSONs kommt die Klasse als Schlüssel "class" (z.B. {"x":..,"y":..,"class":"ery"}).
    Intern wird zusätzlich "c" verwendet, damit ältere Codepfade weiter funktionieren.
    """
    import json as _json

    obj = _json.loads(json_path.read_text(encoding="utf-8"))
    pts = obj.get("points", []) or []
    norm_pts: List[dict] = []
    for p in pts:
        # akzeptiert sowohl "class" als auch das ältere "c"
        cls = p.get("c", None)
        if cls is None:
            cls = p.get("class", None)
        norm_pts.append({"x": p.get("x", 0), "y": p.get("y", 0), "c": cls, "class": cls})
    return obj.get("image", ""), norm_pts


# =========================
# 3 ) Labels: Listen
# =========================
def list_label_jsons(labels_dir: Path) -> List[Path]:
    return sorted(
        [p for p in labels_dir.rglob("*_points.json") if p.is_file()]
    )  # Sortiert alle labels


# =========================
# 4 ) Tiles: Einteilung
# =========================
def _iter_tiles(image_width: int, image_height: int, tile_spec: TileSpec):
    step_x_pixels = tile_spec.tile_w - tile_spec.overlap
    step_y_pixels = tile_spec.tile_h - tile_spec.overlap

    x_positions = list(
        range(0, max(image_width - tile_spec.tile_w, 0) + 1, step_x_pixels)
    ) or [0]
    y_positions = list(
        range(0, max(image_height - tile_spec.tile_h, 0) + 1, step_y_pixels)
    ) or [0]

    if x_positions[-1] != max(image_width - tile_spec.tile_w, 0):
        x_positions.append(max(image_width - tile_spec.tile_w, 0))
    if y_positions[-1] != max(image_height - tile_spec.tile_h, 0):
        y_positions.append(max(image_height - tile_spec.tile_h, 0))

    for y0 in y_positions:
        for x0 in x_positions:
            yield x0, y0, x0 + tile_spec.tile_w, y0 + tile_spec.tile_h


# =========================
# 5 ) Tiles: Liste aus Punkte
# =========================
# Filtert label-Punkte im Tile und gibt sie als Liste zurück
def _points_in_tile(
    points: List[dict], x0: int, y0: int, x1: int, y1: int
) -> List[dict]:

    tile_points: List[dict] = []
    for p in points:
        x, y = float(p.get("x", 0)), float(p.get("y", 0))
        if x0 <= x < x1 and y0 <= y < y1:
            cls = p.get("c") if p.get("c") is not None else p.get("class") if p.get("c") is not None else p.get("class")
            tile_points.append({"x": x - x0, "y": y - y0, "c": cls, "class": cls})
    return tile_points


# =========================
# 6 ) Tile: Zuordnung
# =========================
# Zuordnung der Tiles zu passenden Label-Punkten
def tile_image_and_points(
    image_bgr: np.ndarray, points: List[dict], tile_spec: TileSpec
):

    image_height, image_width = image_bgr.shape[:2]
    tiles = []
    for x0, y0, x1, y1 in _iter_tiles(image_width, image_height, tile_spec):
        tile = image_bgr[y0:y1, x0:x1].copy()
        tile_points = _points_in_tile(points, x0, y0, x1, y1)
        tiles.append((tile, tile_points, (x0, y0, x1, y1)))
    return tiles


# =========================
# 7 ) Tile: Datenvorverarbeitung KI
# =========================
# BGR --> RGB
def preprocess_image_bgr(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


# =========================
# 8 ) Zählung
# =========================
# Zählt Zellen pro Klasse (für Batch-lernen)
def image_cell_count(points: List[dict]) -> int:
    return int(len(points))


# =========================
# 9 ) KI: Zellzentren, gültige Bereiche
# =========================
def generate_targets(
    tile_shape_hw: tuple[int, int],
    tile_points: List[dict],
    sigma_px: Dict[str, float],
    blob_r: Dict[str, int],
):
    # ===== Trainings-Targets =====
    # --> Heatmaps der Zellzentren (pro Klasse) und eine Maske
    tile_h, tile_w = tile_shape_hw

    centers = np.zeros((tile_h, tile_w, len(CLASSES)), dtype=np.float32)
    mask = np.zeros((tile_h, tile_w, 1), dtype=np.float32)

    for p in tile_points:
        cls = p.get("c") if p.get("c") is not None else p.get("class")
        if cls not in CLASSES:
            continue
        cx, cy = float(p["x"]), float(p["y"])
        ci = CLASSES.index(cls)

        s = float(sigma_px[cls])
        x = np.arange(tile_w, dtype=np.float32)
        y = np.arange(tile_h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * (s**2)))
        centers[..., ci] = np.maximum(centers[..., ci], gauss.astype(np.float32))

        r = int(blob_r[cls])
        x0 = max(int(cx) - r, 0)
        x1 = min(int(cx) + r + 1, tile_w)
        y0 = max(int(cy) - r, 0)
        y1 = min(int(cy) + r + 1, tile_h)
        mask[y0:y1, x0:x1, 0] = 1.0

    return centers, mask