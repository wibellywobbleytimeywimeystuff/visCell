"""Modul: config.py
Beschreibung:
Dieses Skript legt die zentralen Konfigurationsklassen und Standardparameter für
Training und Inferenz (z.B. Tile-Größe, Batch-Size, Lernrate und Klassenbezeichnungen) fest

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

CLASSES = ("ery", "hefe", "leuko")


# =========================
# 2 ) Klasse: TileSpec
# =========================
@dataclass(frozen=True)
class TileSpec:

    tile_w: int = 512
    tile_h: int = 512

    overlap: int = 64  # px

    safe_margin: int = 32


# =========================
# 3 ) Klasse: Config
# =========================
@dataclass
class Config:
    data_root: str = "data"

    tile: TileSpec = TileSpec()

    # Glockenförmige Verteilung um das Zentrum (center-heatmap)
    sigma_px_ery: float = 6.0
    sigma_px_hefe: float = 7.0
    sigma_px_leuko: float = 8.0

    # blob = Radius um Zellmarker (jsons) für Zellzentrum
    blob_r_ery: int = 10
    blob_r_hefe: int = 12
    blob_r_leuko: int = 14

    a_max: int = 350
    b_max: int = 1200

    # Einstellungen: trainieren
    batch_size: int = 2
    steps_per_epoch: int = 100
    val_steps: int = 40

    epochs_total: int = 100
    base_filters: int = 16
    depth: int = 4
    dropout: float = 0.1

    learning_rate: float = 2e-4

    output_directory: str = "runs"
    run_name: str = "viscell_unet_512_cpu"
