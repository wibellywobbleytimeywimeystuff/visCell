"""Modul: __init__.py
Beschreibung:
Dieses Paket enthält Konfiguration, Datenstrukturen und (optional) Zähl-/Postprocessing-Funktionen
für die visCell-KI.

Hinweis:
Importe von rechen-/opencv-lastigen Modulen werden bewusst verzögert, damit ein Import von
viscell_ai.config nicht durch Postprocessing-Implementierungen blockiert wird.

Autor: Marlon Aust
Projektname: visCell
"""
from __future__ import annotations

# Re-Exports (leichtgewichtig)
from .config import Config, CLASSES, TileSpec  # noqa: F401
from .counting import CountParams  # noqa: F401

# Zählfunktionen werden lazy importiert, damit "config" immer importierbar bleibt.
def count_from_maps(*args, **kwargs):  # noqa: D401
    """Lazy wrapper um viscell_ai.counting.count_from_maps."""
    from .counting import count_from_maps as _cfm  # local import
    return _cfm(*args, **kwargs)


def safe_crop_maps(*args, **kwargs):
    """Lazy wrapper um viscell_ai.counting.safe_crop_maps."""
    from .counting import safe_crop_maps as _scm  # local import
    return _scm(*args, **kwargs)
