"""
Modul: augmentation.py
Beschreibung:
Dieses Skript führt eine einfache Datenaugmentation durch
(Rotation von Bilddaten um 0°, 90°, 180°, 270°) und speichert
die Ergebnisse in einem Unterordner.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden aller Bibliotheken
# =========================
import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
import os
import threading


# =========================
# 2 ) Hauptschleife
# =========================
class augmentation(ctk.CTk):

    # =========================
    # 3 ) GUI
    # =========================
    def __init__(self):
        super().__init__()
        self.title("Skript Augmentation")
        self.geometry("500x300")

        self.btn_select = ctk.CTkButton(
            self, text="Ordner wählen", command=self.process
        )
        self.btn_select.pack(pady=100)
        self.status_label = ctk.CTkLabel(self, text="Bereit", text_color="gray")
        self.status_label.pack(pady=20)

    # =========================
    # 4 ) Bildmanipulation
    # =========================
    # ===== Processing im Hintergrund =====
    def process(self):
        path = filedialog.askdirectory()
        if not path:
            return
        threading.Thread(target=self.rotate, args=(path,), daemon=True).start() # Anwendung bleibt bedienbar

    # ===== Augmentation =====
    def rotate(self, source_folder):
        output_folder = os.path.join(source_folder, "augmented")
        os.makedirs(output_folder, exist_ok=True)

        # ===== Kompatibel Versionen =====
        extensions = (".jpg", ".jpeg", ".png", ".bmp") # Erlaubte Dateiformate
        files = [f for f in os.listdir(source_folder) if f.lower().endswith(extensions)]

        # ===== Quelle: Ordner =====
        for filename in files:
            img = cv2.imread(os.path.join(source_folder, filename))
            if img is None:  # Kein Bild ausgewählt mit passender Endung
                continue
            name, ext = os.path.splitext(filename)

            # ===== Drehung =====
            # k=0: 0 CW, k=1: 90 CW, k=2: 180 CW, k=3: 270 CW
            for k in range(4):
                rotated = np.rot90(img, k=-k)
                new_name = f"{name}_{k*90}deg{ext}"
                cv2.imwrite(os.path.join(output_folder, new_name), rotated)

        # ===== Endnachricht =====
        self.status_label.configure(
            text="Fertig! Ordner 'augmented' erstellt.", text_color="green"
        )


# =========================
# 5 ) Start der Anwendung
# =========================
if __name__ == "__main__":
    app = augmentation()
    app.mainloop()
