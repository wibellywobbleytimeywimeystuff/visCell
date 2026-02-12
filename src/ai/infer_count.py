"""
Modul: infer_count.py
Beschreibung:
Dieses Skript führt die KI-Inferenz zur Analyse mikroskopischer Zellstrukturen durch.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der KI-Bibliotheken
# =========================
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from viscell_ai.config import Config, CLASSES, TileSpec
from viscell_ai.data import preprocess_image_bgr
from viscell_ai.counting import count_from_maps, CountParams


# =========================
# 2 ) Iter Tiles
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
# 3 ) KI-Inferenz
# =========================
def run_inference(image_path: Path, model_path: Path, config: Config):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Bild konnte nicht gelesen werden: {image_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    image_height, image_width = image_bgr.shape[:2]
    total_counts = {c: 0 for c in CLASSES}

    for x0, y0, x1, y1 in _iter_tiles(image_width, image_height, config.tile):
        tile_bgr = image_bgr[y0:y1, x0:x1]

        if (
            tile_bgr.shape[0] != config.tile.tile_h
            or tile_bgr.shape[1] != config.tile.tile_w
        ):
            padded = np.zeros(
                (config.tile.tile_h, config.tile.tile_w, 3), dtype=tile_bgr.dtype
            )
            padded[: tile_bgr.shape[0], : tile_bgr.shape[1]] = tile_bgr
            tile_bgr = padded

        x = preprocess_image_bgr(tile_bgr)[None, ...]
        pred = model.predict(x, verbose=0)
        centers = pred["centers"][0]
        mask = pred["mask"][0]

        tile_counts = count_from_maps(centers, mask, config.tile, CountParams())
        for k, v in tile_counts.items():
            total_counts[k] += int(v)

    return total_counts


# =========================
# 4 ) Start
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="visCell AI: Inferenz + Zählung auf einem Bild"
    )
    parser.add_argument(
        "--image", required=True, type=Path, help="Pfad zum Eingabebild"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Pfad zum gespeicherten Keras-Modell (.keras/.h5)",
    )
    args = parser.parse_args()

    config = Config()
    counts = run_inference(args.image, args.model, config)
    print(counts)


# =========================
# 5 ) Start der Anwendung
# =========================
if __name__ == "__main__":
    main()
