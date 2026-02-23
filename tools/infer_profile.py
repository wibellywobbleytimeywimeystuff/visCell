"""
infer_profile.py
Profiling tool for visCell inference. Prints per-tile timing for:
- model.predict
- counting/postprocessing (count_from_maps)

Run from project root, e.g.:
  py tools/infer_profile.py --model runs/viscell_unet/model_best.keras --image data/batch_A/stained/A_st_001.png --max_tiles 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

# --- Make sure 'viscell_ai' is importable when running from project root ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../visCell
SRC_AI = PROJECT_ROOT / "src" / "ai"
if str(SRC_AI) not in sys.path:
    sys.path.insert(0, str(SRC_AI))

import cv2  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from viscell_ai.config import Config
from viscell_ai.data import preprocess_image_bgr
from viscell_ai.counting import count_from_maps, CountParams


def iter_tiles(w: int, h: int, tile_h: int, tile_w: int, overlap: int):
    step_x = tile_w - overlap
    step_y = tile_h - overlap
    xs = list(range(0, max(w - tile_w, 0) + 1, step_x)) or [0]
    ys = list(range(0, max(h - tile_h, 0) + 1, step_y)) or [0]
    # ensure last tile covers right/bottom edge
    last_x = max(w - tile_w, 0)
    last_y = max(h - tile_h, 0)
    if xs[-1] != last_x:
        xs.append(last_x)
    if ys[-1] != last_y:
        ys.append(last_y)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, x0 + tile_w, y0 + tile_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--max_tiles", type=int, default=0, help="0 = all tiles; otherwise limit to N tiles")
    args = ap.parse_args()

    cfg = Config()
    tile = cfg.tile

    print(">> read image...", flush=True)
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    h, w = img.shape[:2]
    print(f">> image shape: {h}x{w}", flush=True)

    print(">> load model...", flush=True)
    t0 = time.perf_counter()
    model = tf.keras.models.load_model(str(args.model), compile=False)
    print(f">> model loaded in {time.perf_counter()-t0:.3f}s", flush=True)

    boxes = list(iter_tiles(w, h, tile.tile_h, tile.tile_w, tile.overlap))
    n_tiles = len(boxes)
    if args.max_tiles and args.max_tiles > 0:
        boxes = boxes[: args.max_tiles]
    print(f">> tiles total={n_tiles}, profiling={len(boxes)} (tile={tile.tile_w}x{tile.tile_h}, overlap={tile.overlap})", flush=True)

    params = CountParams()
    total = {"ery": 0, "leuko": 0, "hefe": 0}

    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        tile_bgr = img[y0:y1, x0:x1]
        if tile_bgr.shape[0] != tile.tile_h or tile_bgr.shape[1] != tile.tile_w:
            pad = np.zeros((tile.tile_h, tile.tile_w, 3), dtype=tile_bgr.dtype)
            pad[: tile_bgr.shape[0], : tile_bgr.shape[1]] = tile_bgr
            tile_bgr = pad

        x = preprocess_image_bgr(tile_bgr)[None, ...]

        tp = time.perf_counter()
        pred = model.predict(x, verbose=0)
        t_pred = time.perf_counter() - tp

        tc = time.perf_counter()
        counts = count_from_maps(pred["centers"][0], pred["mask"][0], tile, params)
        t_count = time.perf_counter() - tc

        for k in total:
            total[k] += int(counts.get(k, 0))

        print(f">> tile {i}/{len(boxes)} predict={t_pred:.3f}s count={t_count:.3f}s counts={counts}", flush=True)

    print(">> TOTAL (partial if max_tiles>0):", total, flush=True)


if __name__ == "__main__":
    main()
