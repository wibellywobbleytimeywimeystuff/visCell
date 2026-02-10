"""
Modul: augmentation.py
Beschreibung:
Dieses Skript führt eine einfache Datenaugmentation durch
(Rotation + Spiegelung) und speichert die Ergebnisse
in einem Unterordner. Passende JSON-Labels werden automatisch
aus data/labels_points mitgenommen und passend transformiert.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden aller Bibliotheken
# =========================
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import cv2
import numpy as np


# =========================
# 2 ) Datentypen
# =========================
@dataclass(frozen=True)
class BildInfo:
    pfad: Path
    breite: int
    hoehe: int


# =========================
# 3 ) Hauptschleife
# =========================
class Augmentation(ctk.CTk):

    # =========================
    # 4 ) GUI
    # =========================
    def __init__(self):
        super().__init__()
        self.title("Skript Augmentation")
        self.geometry("520x320")

        # ===== Pfad zu labeled Rohbilder =====
        self.projekt_ordner = (
            Path(__file__).resolve().parents[1]
            if Path(__file__).resolve().parents[1].name == "visCell"
            else Path(__file__).resolve().parent
        )
        self.ordner_raw = self.projekt_ordner / "data" / "raw"
        self.ordner_labels = self.projekt_ordner / "data" / "labels_points"
        self.ordner_labels.mkdir(parents=True, exist_ok=True)

        # ===== Button =====
        self.btn_select = ctk.CTkButton(
            self, text="Bilder auswählen & augmentieren", command=self.process
        )
        self.btn_select.pack(pady=90)

        self.status_label = ctk.CTkLabel(self, text="Bereit", text_color="gray")
        self.status_label.pack(pady=14)

    # =========================
    # 5 ) Verarbeitung im Hintergrund
    # =========================
    def process(self):
        start_ordner = (
            str(self.ordner_raw) if self.ordner_raw.exists() else str(Path.cwd())
        )
        dateien = filedialog.askopenfilenames(
            title="Bilder auswählen",
            initialdir=start_ordner,
            filetypes=[  # Erlaubte Dateiformate
                ("Bilder", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if not dateien:
            self._status("Keine Auswahl.", "gray")
            return
        bilder = [Path(p) for p in dateien]
        # ===== Prozess wird im Hintergrund ausgeführt =====
        threading.Thread(target=self.augmentiere, args=(bilder,), daemon=True).start()

    # =========================
    # 6 ) Augmentation (Rotation + Spiegelung)
    # =========================
    def augmentiere(self, bilder: list[Path]):
        try:
            # ===== Pfad ok, Ordner vorhanden =====
            quell_ordner = bilder[0].parent
            ordner_bilder_aug = quell_ordner / "augmented"
            ordner_bilder_aug.mkdir(parents=True, exist_ok=True)

            ordner_labels_aug = self.ordner_labels / "augmented"
            ordner_labels_aug.mkdir(parents=True, exist_ok=True)

            # ===== Alle Bilder im Pfad laden =====
            for bild_pfad in bilder:
                info, img = self._lade_bild(bild_pfad)
                if img is None:
                    continue

                label_daten = self._lade_label_json(
                    bild_pfad.stem
                )  # lade JSON-Datei(n)

                # ===== Rotation & Spiegelung =====
                # rot_k: 0, 90, 180, 270 (CW)
                for rot_k in (0, 90, 180, 270):
                    self._speichere_variante(
                        info=info,
                        img=img,
                        label_daten=label_daten,
                        ziel_ordner_bild=ordner_bilder_aug,
                        ziel_ordner_label=ordner_labels_aug,
                        variante=("rot", rot_k),
                    )
                    self._speichere_variante(
                        info=info,
                        img=img,
                        label_daten=label_daten,
                        ziel_ordner_bild=ordner_bilder_aug,
                        ziel_ordner_label=ordner_labels_aug,
                        variante=("flip_h_rot", rot_k),  # Spiegeln horizontal
                    )

            self._status(
                "Fertig! Ordner 'augmented' erstellt.", "green"
            )  # green = Farbtext

        # ===== Fehlermeldung =====
        except Exception as ex:
            self._status(f"Fehler: {ex}", "red")  # rot = Farbtext

    # =========================
    # 7 ) Laden / Speichern
    # =========================
    def _lade_bild(self, bild_pfad: Path) -> tuple[BildInfo | None, np.ndarray | None]:
        img = cv2.imread(str(bild_pfad))
        if img is None:
            return None, None
        hoehe, breite = img.shape[:2]
        return BildInfo(pfad=bild_pfad, breite=breite, hoehe=hoehe), img

    def _lade_label_json(self, stem: str) -> dict | None:
        # labelmaker speichert <stem>_points.json in data/labels_points
        pfad = self.ordner_labels / f"{stem}_points.json"
        if not pfad.exists():
            return None
        try:
            return json.loads(pfad.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _speichere_variante(
        self,
        info: BildInfo,
        img: np.ndarray,
        label_daten: dict | None,
        ziel_ordner_bild: Path,
        ziel_ordner_label: Path,
        variante: tuple[str, int],
    ):
        modus, rot = variante
        name = info.pfad.stem
        ext = info.pfad.suffix

        # ===== Rotieren oder rotieren und spiegeln
        match modus:
            case "rot":  # Rotieren
                img_aug, w_neu, h_neu = self._bild_rotieren(
                    img, info.breite, info.hoehe, rot
                )
                suffix = f"_{rot}deg"
            case "flip_h_rot":  # Rotieren und spiegeln
                img_aug, w_neu, h_neu = self._bild_spiegeln_und_rotieren(
                    img, info.breite, info.hoehe, rot
                )
                suffix = f"_flipH_{rot}deg"
            case _:
                return

        # ===== Augmentierte Bilder erstellen =====
        neuer_dateiname = f"{name}{suffix}{ext}"
        cv2.imwrite(str(ziel_ordner_bild / neuer_dateiname), img_aug)

        if label_daten is None:  # Wenn JSON(s) vorhanden
            return

        # ===== Neue JSON(s) erstellen =====
        neue_json = self._transformiere_label_json(
            label_daten=label_daten,
            w_alt=info.breite,
            h_alt=info.hoehe,
            w_neu=w_neu,
            h_neu=h_neu,
            variante=variante,
            neuer_bildname=neuer_dateiname,
        )
        ziel_json = ziel_ordner_label / f"{name}{suffix}_points.json"
        try:
            ziel_json.write_text(
                json.dumps(neue_json, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass  # weiter

    # =========================
    # 8 ) Bild-Transformationen
    # =========================
    def _bild_rotieren(
        self, img: np.ndarray, w: int, h: int, rot_cw: int
    ) -> tuple[np.ndarray, int, int]:
        rot_cw = rot_cw % 360
        match rot_cw:  # Drehungen
            case 0:
                return img.copy(), w, h
            case 90:
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), h, w
            case 180:
                return cv2.rotate(img, cv2.ROTATE_180), w, h
            case 270:
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), h, w
            case _:
                return img.copy(), w, h

    def _bild_spiegeln_und_rotieren(
        self, img: np.ndarray, w: int, h: int, rot_cw: int
    ) -> tuple[np.ndarray, int, int]:
        gespiegelt = cv2.flip(img, 1)  # Spiegeln
        return self._bild_rotieren(gespiegelt, w, h, rot_cw)  # dann spiegeln

    # =========================
    # 9 ) Label-Transformationen
    # =========================
    def _transformiere_label_json(
        self,
        label_daten: dict,
        w_alt: int,
        h_alt: int,
        w_neu: int,
        h_neu: int,
        variante: tuple[str, int],
        neuer_bildname: str,
    ) -> dict:
        modus, rot = variante
        rot = rot % 360

        punkte_alt = label_daten.get("points", [])
        punkte_neu = []

        for p in punkte_alt:
            try:
                x = float(p["x"])
                y = float(p["y"])
                klasse = p.get("class", "")
            except Exception:
                continue

            match modus:
                case "rot":
                    x2, y2 = self._punkt_rotieren_cw(x, y, w_alt, h_alt, rot)
                case "flip_h_rot":
                    x1, y1 = self._punkt_spiegeln_h(x, y, w_alt)
                    x2, y2 = self._punkt_rotieren_cw(x1, y1, w_alt, h_alt, rot)
                case _:
                    continue

            # runden wie im Labelmaker (aber etwas großzügig)
            punkte_neu.append({"x": round(x2, 3), "y": round(y2, 3), "class": klasse})

        return {
            "image": neuer_bildname,
            "points": punkte_neu,
            "meta": {
                "source_image": label_daten.get("image", ""),
                "augmentation": {
                    "mode": modus,
                    "rotation_cw_deg": rot,
                    "new_size": [w_neu, h_neu],
                },
            },
        }

    def _punkt_spiegeln_h(self, x: float, y: float, w: int) -> tuple[float, float]:
        return (w - 1) - x, y  # Spiegelung an der vertikalen Achse

    def _punkt_rotieren_cw(
        self, x: float, y: float, w: int, h: int, rot_cw: int
    ) -> tuple[float, float]:

        # Rotation um Bildursprung
        match rot_cw:
            case 0:  # 0°: (x,y)
                return x, y
            case 90:  # 90° CW
                return (h - 1) - y, x
            case 180:  # 180° (CW)
                return (w - 1) - x, (h - 1) - y
            case 270:  # 270° CW
                return y, (w - 1) - x
            case _:
                return x, y

    # =========================
    # 10 ) Status
    # =========================
    def _status(self, text: str, farbe: str):
        # Update GUI
        self.after(0, lambda: self.status_label.configure(text=text, text_color=farbe))


# =========================
# 11 ) Start der Anwendung
# =========================
if __name__ == "__main__":
    app = Augmentation()
    app.mainloop()
