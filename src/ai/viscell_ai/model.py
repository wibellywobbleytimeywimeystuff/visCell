"""
Modul: model.py
Beschreibung:
Dieses Skript definiert die Modellarchitektur (U-Net) zur Vorhersage von
Zellzentrum-Heatmaps („centers“) und einer gültigen Bildmaske („mask“) auf
Bildkacheln (Tiles). Das Modell besitzt Skip-Connections und erzeugt zwei
Ausgabe-Tensoren (centers: 3 Kanäle, mask: 1 Kanal), jeweils mit Sigmoid-
Aktivierung.

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

L = tf.keras.layers


# =========================
# 2 ) Convolutional
# =========================
# Zwei Conv-Layer (3x3) mit BatchNorm + ReLU
def _conv_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> tf.Tensor:

    x = L.Conv2D(filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Conv2D(filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    if dropout and dropout > 0:
        x = L.SpatialDropout2D(dropout)(x)

    return x


# =========================
# 3 ) Struktur: U-Net
# =========================
# Outputs:
# - "centers": 3 Heatmap-Kanäle (z.B. ery/hefe/leuko) mit Sigmoid
# - "mask":    1 Kanal (gültiger Bereich) mit Sigmoid
def build_unet(
    input_shape: tuple[int, int, int] = (1024, 1024, 3),
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.1,
) -> tf.keras.Model:

    # Eingabe: Tile
    image_input = L.Input(shape=input_shape, name="image")

    x = image_input
    skip_connections: list[tf.Tensor] = []

    # =========================
    # 3.1 ) Encoder (Downsampling)
    # =========================
    # Encoder: Detail-Extraktion + MaxPooling, Filterzahl steigt pro Stufe
    filters = base_filters
    for level in range(depth):
        x = _conv_block(x, filters, dropout=dropout if level > 0 else 0.0)
        skip_connections.append(x)

        # Downsampling: Auflösung halbieren, Featurezahl steigt
        x = L.MaxPool2D(2)(x)
        filters *= 2

    # =========================
    # 3.2 ) Bottleneck (tiefster Punkt)
    # =========================
    x = _conv_block(x, filters, dropout=dropout)

    # =========================
    # 3.3 ) Decoder (Upsampling)
    # =========================
    for level in reversed(range(depth)):
        filters //= 2

        # Upsampling: Auflösung verdoppeln
        x = L.UpSampling2D(2, interpolation="bilinear")(x)

        # Skip-Connection: Encoder-Features gleicher Auflösung
        x = L.Concatenate()([x, skip_connections[level]])

        # Conv-Block zur Merkmalsverarbeitung
        x = _conv_block(x, filters, dropout=dropout)

    # =========================
    # 3.4 ) Output-Köpfe (Heads)
    # =========================
    # centers_out: 3 Kanäle (Heatmaps pro Zelltyp)
    centers_out = L.Conv2D(3, 1, activation="sigmoid", name="centers")(x)

    # mask_out: 1 Kanal (gültige Bereiche)
    mask_out = L.Conv2D(1, 1, activation="sigmoid", name="mask")(x)

    # Modell mit zwei Outputs zurückgeben
    return tf.keras.Model(
        inputs=image_input,
        outputs={"centers": centers_out, "mask": mask_out},
        name="viscell_unet",
    )
