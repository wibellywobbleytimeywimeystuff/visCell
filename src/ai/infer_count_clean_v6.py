
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

CLASSES = ("ery", "hefe", "leuko")
IDX = {c:i for i,c in enumerate(CLASSES)}

@dataclass(frozen=True)
class TileSpec:
    tile: int = 512
    overlap: int = 64

@dataclass(frozen=True)
class Peak:
    x: int
    y: int
    cls: str
    conf: float 

def _iter_tiles(w: int, h: int, spec: TileSpec) -> Iterable[Tuple[int, int, int, int]]:
    step = spec.tile - spec.overlap
    xs = list(range(0, max(w - spec.tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - spec.tile, 0) + 1, step)) or [0]
    last_x = max(w - spec.tile, 0)
    last_y = max(h - spec.tile, 0)
    if xs[-1] != last_x: xs.append(last_x)
    if ys[-1] != last_y: ys.append(last_y)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, x0 + spec.tile, y0 + spec.tile

def _preprocess(tile_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)[None, ...]

def _extract_centers(pred) -> np.ndarray:
    if isinstance(pred, dict):
        return pred["centers"][0]
    if isinstance(pred, (list, tuple)):
        arr = pred[0]
        return arr[0] if hasattr(arr, "__getitem__") and getattr(arr, "ndim", 0) >= 3 else arr
    return pred[0] if getattr(pred, "ndim", 0) == 4 else pred

def _robust_thr(hm: np.ndarray, quantile: float, abs_thresh: float, max_fg: float) -> float:
    hm_f = hm.astype(np.float32)
    mx = float(hm_f.max()) if hm_f.size else 0.0
    if mx <= 1e-8:
        return 1.0
    for q in (quantile, 0.997, 0.999, 0.9995):
        thr = float(np.quantile(hm_f, q))
        thr = max(abs_thresh, thr)
        fg = float((hm_f >= thr).mean())
        if fg <= max_fg:
            return thr
    thr = float(np.quantile(hm_f, quantile))
    return max(abs_thresh, thr)

def _peak_mask(hm: np.ndarray, thr: float, detect_dist: int) -> np.ndarray:
    k = max(3, int(detect_dist) * 2 + 1)
    kernel = np.ones((k, k), np.uint8)
    hm_f = hm.astype(np.float32)
    dil = cv2.dilate(hm_f, kernel)
    return (hm_f >= thr) & (hm_f == dil)

def _components_centroids(mask: np.ndarray, max_area: int) -> List[Tuple[int, int, int]]:
    u8 = (mask.astype(np.uint8) * 255)
    n, _, stats, centroids = cv2.connectedComponentsWithStats((u8 > 0).astype(np.uint8), connectivity=8)
    out: List[Tuple[int, int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 0 < area <= max_area:
            cx, cy = centroids[i]
            out.append((int(round(cx)), int(round(cy)), area))
    return out

def _nms_by_distance(peaks: List[Peak], merge_dist: int) -> List[Peak]:
    if not peaks: return []
    peaks_sorted = sorted(peaks, key=lambda p: p.conf, reverse=True)
    keep: List[Peak] = []
    min_d2 = float(merge_dist * merge_dist)
    for p in peaks_sorted:
        ok = True
        for k in keep:
            dx = float(p.x - k.x)
            dy = float(p.y - k.y)
            if dx*dx + dy*dy < min_d2:
                ok = False
                break
        if ok:
            keep.append(p)
    return keep

def _pick_class(
    vec3: np.ndarray,
    class_weights: np.ndarray,
    class_margin: float,
    ambiguous_policy: str,
    leuko_min_abs: float,
    leuko_min_rel: float,
    hefe_min_abs: float,
    hefe_min_rel: float,
    leuko_margin_over_ery: float,
    thr_tile: float,
) -> Tuple[str, str]:
    """
    returns (picked_class, reason)
    reason in {"ok","margin","leuko_guard","hefe_guard"}
    """
    v_raw = vec3.astype(np.float32)
    v = v_raw * class_weights.astype(np.float32)

    idx = np.argsort(v)[::-1]
    top = int(idx[0]); second = int(idx[1])
    top1 = float(v[top]); top2 = float(v[second])
    if (top1 - top2) < float(class_margin):
        return ("" if ambiguous_policy == "drop" else "ery", "margin")

    leuko_min_eff = max(float(leuko_min_abs), float(leuko_min_rel) * float(thr_tile))
    hefe_min_eff  = max(float(hefe_min_abs),  float(hefe_min_rel)  * float(thr_tile))

    if top == IDX["leuko"]:
        if float(v_raw[IDX["leuko"]]) < leuko_min_eff:
            return ("ery", "leuko_guard")
        if (float(v_raw[IDX["leuko"]]) - float(v_raw[IDX["ery"]])) < float(leuko_margin_over_ery):
            return ("ery", "leuko_guard")
    if top == IDX["hefe"]:
        if float(v_raw[IDX["hefe"]]) < hefe_min_eff:
            return ("ery", "hefe_guard")

    return (CLASSES[top], "ok")

def infer_and_count(
    model_path: Path,
    image_path: Path,
    spec: TileSpec,
    quantile: float,
    abs_thresh: float,
    max_fg: float,
    detect_dist: int,
    merge_dist: int,
    max_area: int,
    peak_rel: float,
    class_weights: np.ndarray,
    class_margin: float,
    ambiguous_policy: str,
    leuko_min_abs: float,
    leuko_min_rel: float,
    hefe_min_abs: float,
    hefe_min_rel: float,
    leuko_margin_over_ery: float,
    debug: bool,
) -> Dict[str, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Kann Bild nicht lesen: {image_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    h, w = img.shape[:2]
    all_peaks: List[Peak] = []
    tiles = list(_iter_tiles(w, h, spec))

    raw_wins = {c: 0 for c in CLASSES}
    forced = {"margin": 0, "leuko_guard": 0, "hefe_guard": 0}

    if debug:
        print(f"[DEBUG] image={w}x{h}, tiles={len(tiles)}, tile={spec.tile}, overlap={spec.overlap}")
        print(f"[DEBUG] weights={class_weights.tolist()} class_margin={class_margin} ambiguous_policy={ambiguous_policy}")
        print(f"[DEBUG] leuko_min_abs={leuko_min_abs} leuko_min_rel={leuko_min_rel} hefe_min_abs={hefe_min_abs} hefe_min_rel={hefe_min_rel} leuko_margin_over_ery={leuko_margin_over_ery}")

    for ti, (x0, y0, x1, y1) in enumerate(tiles, start=1):
        tile = img[y0:y1, x0:x1]
        if tile.shape[0] != spec.tile or tile.shape[1] != spec.tile:
            pad = np.zeros((spec.tile, spec.tile, 3), dtype=tile.dtype)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tile = pad

        pred = model.predict(_preprocess(tile), verbose=0)
        centers = _extract_centers(pred)
        centers3 = centers[..., :3].astype(np.float32)
        center_max = np.max(centers3, axis=-1)

        thr = _robust_thr(center_max, quantile=quantile, abs_thresh=abs_thresh, max_fg=max_fg)
        pmask = _peak_mask(center_max, thr=thr, detect_dist=detect_dist)
        if float(pmask.mean()) > 0.02:
            continue
        cents = _components_centroids(pmask, max_area=max_area)

        if debug and ti <= 2:
            print(f"[DEBUG] tile#{ti} thr={thr:.6f} detect_peaks={len(cents)} center_max.max={float(center_max.max()):.6f}")

        for cx, cy, _ in cents:
            cy0 = int(np.clip(cy, 0, centers3.shape[0]-1))
            cx0 = int(np.clip(cx, 0, centers3.shape[1]-1))
            peak_strength = float(center_max[cy0, cx0])
            if peak_strength < float(thr) * float(peak_rel):
                continue

            vec = centers3[cy0, cx0, :]
            widx = int(np.argmax(vec * class_weights))
            raw_wins[CLASSES[widx]] += 1

            picked, reason = _pick_class(
                vec, class_weights, class_margin, ambiguous_policy,
                leuko_min_abs, leuko_min_rel, hefe_min_abs, hefe_min_rel,
                leuko_margin_over_ery, thr_tile=thr
            )
            if reason != "ok":
                forced[reason] += 1
            if picked == "":
                continue
            all_peaks.append(Peak(x=x0 + cx, y=y0 + cy, cls=picked, conf=peak_strength))

    if debug:
        print(f"[DEBUG] raw winners (weighted): {raw_wins}")
        print(f"[DEBUG] forced reasons: {forced}")
        print(f"[DEBUG] peaks before NMS: {len(all_peaks)}")

    merged = _nms_by_distance(all_peaks, merge_dist=merge_dist)

    if debug:
        print(f"[DEBUG] peaks after  NMS: {len(merged)}")

    totals = {c: 0 for c in CLASSES}
    for p in merged:
        totals[p.cls] += 1
    return totals

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)

    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)

    ap.add_argument("--quantile", type=float, default=0.9943)
    ap.add_argument("--abs_thresh", type=float, default=0.0)
    ap.add_argument("--max_fg", type=float, default=0.01)

    ap.add_argument("--detect_dist", type=int, default=6)
    ap.add_argument("--merge_dist", type=int, default=14)

    ap.add_argument("--max_area", type=int, default=120)
    ap.add_argument("--peak_rel", type=float, default=1.0)

    ap.add_argument("--class_weights", type=str, default="1,0.3,1.2")
    ap.add_argument("--class_margin", type=float, default=0.02)
    ap.add_argument("--ambiguous_policy", choices=["ery", "drop"], default="ery")

    ap.add_argument("--leuko_min_abs", type=float, default=0.06)
    ap.add_argument("--leuko_min_rel", type=float, default=0.55)
    ap.add_argument("--hefe_min_abs", type=float, default=0.08)
    ap.add_argument("--hefe_min_rel", type=float, default=0.60)
    ap.add_argument("--leuko_margin_over_ery", type=float, default=0.015)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    w = np.array([float(x.strip()) for x in args.class_weights.split(",")], dtype=np.float32)
    if w.size != 3:
        raise ValueError("--class_weights muss 3 Werte haben, z.B. '1,0.3,1.2'")

    counts = infer_and_count(
        model_path=args.model,
        image_path=args.image,
        spec=TileSpec(tile=args.tile, overlap=args.overlap),
        quantile=args.quantile,
        abs_thresh=args.abs_thresh,
        max_fg=args.max_fg,
        detect_dist=args.detect_dist,
        merge_dist=args.merge_dist,
        max_area=args.max_area,
        peak_rel=args.peak_rel,
        class_weights=w,
        class_margin=float(args.class_margin),
        ambiguous_policy=str(args.ambiguous_policy),
        leuko_min_abs=float(args.leuko_min_abs),
        leuko_min_rel=float(args.leuko_min_rel),
        hefe_min_abs=float(args.hefe_min_abs),
        hefe_min_rel=float(args.hefe_min_rel),
        leuko_margin_over_ery=float(args.leuko_margin_over_ery),
        debug=bool(args.debug),
    )
    print(counts)

if __name__ == "__main__":
    main()
