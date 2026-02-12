"""
Modul: config.py
Beschreibung:
Dieses Skript enthält Konfigurationslogik und Standardparameter der Anwendung.

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



@dataclass(frozen=True)
# =========================
# 2 ) Klasse TileSpec
# =========================

class TileSpec:
    

    tile_w: int = 1024
    tile_h: int = 1024

    overlap: int = 128

    safe_margin: int = 64



@dataclass
# =========================
# 3 ) Klasse Config
# =========================

class Config:
    

    data_root: str = "data"

    tile: TileSpec = TileSpec()

    sigma_px_ery: float = 6.0
    sigma_px_hefe: float = 7.0
    sigma_px_leuko: float = 8.0

    blob_r_ery: int = 10
    blob_r_hefe: int = 12
    blob_r_leuko: int = 14

    a_max: int = 350
    b_max: int = 1200

    batch_size: int = 2
    steps_per_epoch: int = 200
    val_steps: int = 40

    epochs_total: int = 100
    base_filters: int = 32
    depth: int = 4
    dropout: float = 0.1

    learning_rate: float = 2e-4

    output_directory: str = "runs"
    run_name: str = "viscell_unet"