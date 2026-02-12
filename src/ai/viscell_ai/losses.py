"""Modul: losses.py
Beschreibung:
Dieses Skript stellt spezielle Verlustfunktionen (Fehlerberechnung) für das Training bereit, zur robusten Optimierung bei viel Hintergrund, wenige Zellzentren

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der Bibliotheken
# =========================
from __future__ import annotations

import tensorflow as tf


# =========================
# 2 ) Vorhersage: Fehlerreduktion
# =========================
# Berechnung der Losses (Fehler) der Vorhersage
def focal_bce(gamma: float = 2.0, alpha: float = 0.25):

    def _loss(ground_truth, prediction):
        ground_truth = tf.cast(ground_truth, tf.float32)
        prediction = tf.clip_by_value(tf.cast(prediction, tf.float32), 1e-6, 1.0 - 1e-6)

        ce = tf.keras.backend.binary_crossentropy(ground_truth, prediction)

        p_t = ground_truth * prediction + (1.0 - ground_truth) * (1.0 - prediction)
        modulating = tf.pow(1.0 - p_t, gamma)

        focal_loss = alpha * modulating * ce
        return tf.reduce_mean(focal_loss)

    return _loss
