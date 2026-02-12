"""
Modul: train.py
Beschreibung:
Dieses Skript stellt Trainingsroutinen für das verwendete KI-Modell bereit.
Es werden augmentierte Bilder und die dazugehörigen JSONs geladen und trainiert.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der Bibliotheken
# =========================
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf

from viscell_ai.config import Config
from viscell_ai.data import (
    load_points_json,
    list_label_jsons,
    tile_image_and_points,
    preprocess_image_bgr,
    generate_targets,
    image_cell_count,
)
from viscell_ai.model import build_unet
from viscell_ai.losses import focal_bce


# =========================
# 2 ) Trainingsbatches
# =========================
def split_curriculum(
    label_json_paths: List[Path], config: Config
) -> Dict[str, List[Path]]:

    group_a: List[Path] = []  # Batch A
    group_b: List[Path] = []  # Batch B
    group_c: List[Path] = []  # Batch C

    # ===== Lade JSON-Infos =====
    for json_path in label_json_paths:
        _, points = load_points_json(json_path)
        n_cells = image_cell_count(points)

        if n_cells <= config.a_max:
            group_a.append(json_path)
        elif n_cells <= config.b_max:
            group_b.append(json_path)
        else:
            group_c.append(json_path)

    return {"A": group_a, "B": group_b, "C": group_c}


# =========================
# 3 ) Hilfsfunktion: Bild laden wenn Pfad vorhanden
# =========================
def _safe_imread(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return cv2.imread(str(path))  # Bild lesen


# =========================
# 4 ) Bild laden
# =========================
def try_read_image(images_dir: Path, image_name: str) -> Optional[np.ndarray]:

    image = _safe_imread(images_dir / image_name)
    if image is not None:
        return image

    image = _safe_imread(images_dir / "augmented" / image_name)
    if image is not None:
        return image

    stem = Path(image_name).stem
    for folder in (images_dir, images_dir / "augmented"):
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):  # Erlaubte Dateiformate
            image = _safe_imread(folder / f"{stem}{ext}")
            if image is not None:
                return image
    return None


# =========================
# 5 ) Preprocessing: Vorbereitung fürs Trainieren
# =========================
def make_generator(label_jsons: List[Path], images_dir: Path, config: Config):

    # ===== Gaussian-Blur-Werte pro Zelltyp =====
    # Glockenförmige Verteilung um das Zentrum (center-heatmap)
    sigma_px = {
        "ery": config.sigma_px_ery,
        "hefe": config.sigma_px_hefe,
        "leuko": config.sigma_px_leuko,
    }
    # ===== Ziel-Blobs für jeden Zelltyp =====
    # blob = Radius um Zellmarker (jsons) für Zellzentrum
    blob_r = {
        "ery": config.blob_r_ery,
        "hefe": config.blob_r_hefe,
        "leuko": config.blob_r_leuko,
    }
    tile_spec = config.tile  # Tile-Größe

    # ===== Liefert der KI Daten =====
    def gen():
        rng = np.random.default_rng(123)  # Zufallsgenerator

        while True:  # True bis keras stoppt
            rng.shuffle(label_jsons)  # JSONs zufällig laden
            for json_path in label_jsons:
                image_name, points = load_points_json(json_path)

                image_bgr = try_read_image(images_dir, image_name)  # Bilder laden
                if image_bgr is None:
                    continue

                img_h, img_w = image_bgr.shape[:2]
                # ===== Bild < Tile: mit schwarzen Pixeln auffüllen =====
                if img_h < tile_spec.tile_h or img_w < tile_spec.tile_w:
                    pad_h = max(img_h, tile_spec.tile_h)
                    pad_w = max(img_w, tile_spec.tile_w)
                    padded = np.zeros((pad_h, pad_w, 3), dtype=image_bgr.dtype)
                    padded[:img_h, :img_w] = image_bgr
                    image_bgr = padded

                # ===== Bild zu Tiles, Zellpunkte neu berechnet =====
                tiles = tile_image_and_points(image_bgr, points, tile_spec)
                rng.shuffle(tiles)  # Tiles mischen

                # ===== Normalisieren (0...1), BGR zu RGB =====
                for tile_bgr, tile_points, _tile_bbox in tiles:
                    x = preprocess_image_bgr(tile_bgr)

                    # ===== Center-Heatmap erstellen =====
                    centers, mask = generate_targets(  # 3 Kanäle
                        (tile_spec.tile_h, tile_spec.tile_w),
                        tile_points,
                        sigma_px,
                        blob_r,
                    )
                    yield x, {"centers": centers, "mask": mask}  # Ausgabe ans Modell

    return gen


# =========================
# 6 ) Batch: Reihenfolge
# =========================
# Anzahl Epochen, Batchtyp A, Batchtyp B, Batchtyp C
def curriculum_schedule():
    # Pro run immer mehr Batchtyp C, weniger Batchtyp A
    return [
        (15, {"A": 0.8, "B": 0.2, "C": 0.0}),  # not-so-finetuning
        (25, {"A": 0.3, "B": 0.5, "C": 0.2}),
        (45, {"A": 0.1, "B": 0.3, "C": 0.6}),
        (15, {"A": 0.2, "B": 0.3, "C": 0.5}),  # finetuning
    ]


# =========================
# 7 ) Aufbau Datasets
# =========================
# Dataset pro Batchtyp A,B,C
# Output-labels als dict
def build_datasets(groups: Dict[str, List[Path]], images_dir: Path, config: Config):
    output_signature = (
        tf.TensorSpec(
            shape=(config.tile.tile_h, config.tile.tile_w, 3), dtype=tf.float32
        ),
        {
            "centers": tf.TensorSpec(
                shape=(config.tile.tile_h, config.tile.tile_w, 3), dtype=tf.float32
            ),
            "mask": tf.TensorSpec(
                shape=(config.tile.tile_h, config.tile.tile_w, 1), dtype=tf.float32
            ),
        },
    )

    # ===== Datasets pro Batch =====
    datasets = {}
    for key in ("A", "B", "C"):
        if not groups[key]:
            datasets[key] = None
            continue

        # ===== Trainingsdaten erzeugen =====
        # Erzeugt Trainingsdaten in Form von Tiles sowie Heatmaps für Zellmitte
        gen = make_generator(groups[key], images_dir, config)
        dataset = tf.data.Dataset.from_generator(
            gen, output_signature=output_signature
        )  # Tile Input-Bild

        dataset = (
            dataset.shuffle(256)  # Mischt Beispiele → weniger Overfitting
            .batch(config.batch_size)
            .prefetch(tf.data.AUTOTUNE)  # Lädt Batches im Voraus → schnelleres Training
        )
        datasets[key] = dataset

    return datasets


# =========================
# 8 ) Datasets mischen
# =========================
# Erzeugt ein Dataset aus Batch A,B,C - entsprechend der Gewichtung
def mix_datasets(datasets: Dict[str, tf.data.Dataset], weights: Dict[str, float]):

    parts = []  # Datasets
    w = []  # Wahrscheinlichkeiten (Gewichte)
    for k, prob in weights.items():
        if prob <= 0:
            continue
        if datasets.get(k) is None:
            continue
        parts.append(datasets[k])
        w.append(prob)

    if not parts:  # Abbrechen des Trainings, wenn weight = 0
        raise ValueError("Keine Datasets zum Mischen gefunden (weights/labels prüfen).")

    return tf.data.Dataset.sample_from_datasets(parts, weights=w)  # Neues Dataset


# =========================
# 9 ) Training
# =========================
# Lädt labels --> erstellt datasets --> erstellt U-Net Modell --> trainiert
def train(config: Config, data_root: Path):
    labels_dir = data_root / "labels_points"
    images_dir = data_root / "raw"

    # ===== Labels: prüfen =====
    label_jsons = list_label_jsons(labels_dir)
    if not label_jsons:  # Fehlermeldung
        raise FileNotFoundError(f"Keine *_points.json in {labels_dir}")

    curriculum_groups = split_curriculum(
        label_jsons, config
    )  # Daten sortieren nach Batchtyp A,B,C
    datasets = build_datasets(curriculum_groups, images_dir, config)  # Baut tf-Datasets

    # ===== U-Net Modell erzeugen =====
    model = build_unet(
        input_shape=(config.tile.tile_h, config.tile.tile_w, 3),
        base_filters=config.base_filters,
        depth=config.depth,
        dropout=config.dropout,
    )

    # ===== Kompilierer =====
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={
            "centers": focal_bce(gamma=2.0, alpha=0.25),  # Output: center-heat
            "mask": tf.keras.losses.BinaryCrossentropy(),  # Output: mask
        },
        loss_weights={"centers": 1.0, "mask": 0.5},
    )

    # ===== Ausgabe =====
    output_directory = Path(config.output_directory) / config.run_name
    output_directory.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_directory / "model_best.keras"  # KI-Modell

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="loss",
            save_best_only=True,  # Speichert Modell, wenn Gesamt-Loss besser wird
            save_weights_only=False,
        ),
        tf.keras.callbacks.CSVLogger(
            str(output_directory / "train_log.csv"), append=True  # Trainingslog
        ),
    ]

    # ==== Training: loop =====
    # Pro Phase: neue Batchtyp-Mischung
    for phase_epochs, weights in curriculum_schedule():
        train_ds = mix_datasets(datasets, weights)
        model.fit(
            train_ds,
            epochs=phase_epochs,
            steps_per_epoch=config.steps_per_epoch,  # Schritte
            callbacks=callbacks,
        )

    model.save(str(output_directory / "model_final.keras"))  # Modell speichern
    print(f"Training fertig. Modelle liegen in: {output_directory}")


# =========================
# 10 ) Start
# =========================
# Erlaubt training über die Eingabeaufforderung (cmd)
def main():
    parser = argparse.ArgumentParser(
        description="visCell AI Training"
    )  # Aktiviert cmd-Argumente
    parser.add_argument(
        "--data", type=Path, default=Path("data"), help="data_root Ordner"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Run-Name für Output Ordner"
    )
    args = parser.parse_args()

    config = Config()  # Config erstellen/laden
    if args.run_name:
        config.run_name = args.run_name

    train(config, args.data)  #  Training starten


# =========================
# 11 ) Start der Anwendung
# =========================
if __name__ == "__main__":
    main()
