"""
Modul: model.py
Beschreibung:
Dieses Skript enthält Kernlogik zur Ausführung und Orchestrierung der Analysepipeline.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der KI-Bibliotheken
# =========================

from __future__ import annotations




import tensorflow as tf
from tensorflow.keras import layers as L



# =========================
# 2 ) Conv Block
# =========================

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
# 3 ) Aufbau Unet
# =========================

def build_unet(
    input_shape: tuple[int, int, int] = (1024, 1024, 3),
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.1,
) -> tf.keras.Model:
    
    image_input = L.Input(shape=input_shape, name="image")

    x = image_input
    skip_connections: list[tf.Tensor] = []

    filters = base_filters
    for level in range(depth):
        x = _conv_block(x, filters, dropout=dropout if level > 0 else 0.0)
        skip_connections.append(x)
        x = L.MaxPool2D(2)(x)
        filters *= 2

    x = _conv_block(x, filters, dropout=dropout)

    for level in reversed(range(depth)):
        filters //= 2
        x = L.UpSampling2D(2, interpolation="bilinear")(x)
        x = L.Concatenate()([x, skip_connections[level]])
        x = _conv_block(x, filters, dropout=dropout)

    centers_out = L.Conv2D(3, 1, activation="sigmoid", name="centers")(x)
    mask_out = L.Conv2D(1, 1, activation="sigmoid", name="mask")(x)

    return tf.keras.Model(inputs=image_input, outputs={"centers": centers_out, "mask": mask_out}, name="viscell_unet")