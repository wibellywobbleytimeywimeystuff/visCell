"""
Modul: labelmaker.py
Beschreibung:
Dieses Skript dient dem Labeln von Rohbildern, die dann einer KI zum Trainieren gegeben werden können.
Gespeichert werden die Informationen als JSON-Datei. Unterschieden werden:
- Erythrozyten
- Leukozyten
- Hefezellen

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der Bibliotheken
# =========================
import json
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox  # Windows Pop-Up
from PIL import Image, ImageTk


# =========================
# 2 ) Grundeinstellungen
# =========================
# Klassen, die markiert werden können
KLASSEN = ["ery", "leuko", "hefe"]
KLASSEN_TEXT = {
    "ery": "Erythrozyt",
    "leuko": "Leukozyt",
    "hefe": "Hefe",
}

# Farbzuordnung
FARBEN = {
    "ery": "#ff4d4d",
    "leuko": "#4da6ff",
    "hefe": "#ffd24d",
}

PUNKT_RADIUS = 6  # Marker-Größe
LOESCH_RADIUS_SCREEN = 12  # Marker-Größe zum Löschen
DRAG_SCHWELLE = 5  # Threshold klicken vs. ziehen (Mausbewegung)


# =========================
# 3 ) Hilfsfunktionen
# =========================
def ist_bilddatei(datei: Path) -> bool:
    return datei.suffix.lower() in [  # Erlaubte Dateiformate
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    ]


# ===== Zoom-Begrenzung =====
# Verhindert unendliches zoomen
def begrenze(wert: float, min_wert: float, max_wert: float) -> float:
    return max(min_wert, min(max_wert, wert))


# =========================
# 4 ) GUI: Grundeinstellungen
# =========================
class LabelMaker(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ===== Fenster erstellen =====
        self.title("visCell – Labelmaker v2 (einfach)")
        self.geometry("1280x720")  # Fenster Normalgröße
        self.maxsize(1920, 1080)  # Fenster Maximalgröße
        self.minsize(1000, 600)  # Fenster Mindestgröße

        # ===== Pfade =====
        # Annahme: Skript liegt im gleichen Ordner
        self.projekt_ordner = Path(__file__).resolve().parents[2]  # Projekt-Root
        self.ordner_bilder = self.projekt_ordner / "data"
        self.ordner_labels_root = self.projekt_ordner / "data" / "labels_points"
        self.ordner_labels_root.mkdir(parents=True, exist_ok=True)

        # ===== Zustände setzen =====
        self.bilder_liste = []
        self.bild_index = 0
        self.aktuelle_klasse = "ery"  # Vorauswahl

        self.punkte = []  # {"x": float, "y": float, "class": str}

        self.bild_pil = None  # Aktuelles Bild
        self.bild_tk = None  # Anzeige

        # ===== Zoom & ziehen =====
        self.zoom = 1.0
        self.zoom_min = 0.2
        self.zoom_max = 8.0
        self.verschiebung_x = 0.0
        self.verschiebung_y = 0.0

        self._maus_start = None
        self._pan_start = None
        self._hat_gezogen = False  # Ziehen (Mausbewegung)

        # ===== GUI erzeugen =====
        self._ui_bauen()
        self._events_binden()

        self.status("Ordner wählen und labeln ...")

    # =========================
    # 5 ) GUI: erstellen
    # =========================
    def _ui_bauen(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ===== Panel: links =====
        self.panel_links = ctk.CTkFrame(self, width=280)
        self.panel_links.grid(row=0, column=0, sticky="nsw", padx=10, pady=10)

        ctk.CTkLabel(
            self.panel_links,
            text="Labeling von Rohbildern",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        # Button: "Bild-Ordner öffnen"
        self.btn_ordner = ctk.CTkButton(
            self.panel_links, text="Bild-Ordner öffnen", command=self.ordner_auswaehlen
        )
        self.btn_ordner.grid(row=1, column=0, padx=12, pady=6, sticky="ew")

        self.lbl_ordner = ctk.CTkLabel(
            self.panel_links,
            text="(kein Ordner geladen)",
            wraplength=250,
            justify="left",
        )
        self.lbl_ordner.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="w")

        # ===== Auswahl: Klasse wählen =====
        ctk.CTkLabel(
            self.panel_links,
            text="Klasse wählen",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=3, column=0, padx=12, pady=(6, 4), sticky="w")
        self.var_klasse = tk.StringVar(value=self.aktuelle_klasse)

        # Erythrozyten
        self.rb_ery = ctk.CTkRadioButton(
            self.panel_links,
            text="Erythrozyt",
            variable=self.var_klasse,
            value="ery",
            command=self.klasse_geaendert,
        )
        # Leukozyten
        self.rb_leu = ctk.CTkRadioButton(
            self.panel_links,
            text="Leukozyt",
            variable=self.var_klasse,
            value="leuko",
            command=self.klasse_geaendert,
        )
        # Hefezellen
        self.rb_hef = ctk.CTkRadioButton(
            self.panel_links,
            text="Hefe",
            variable=self.var_klasse,
            value="hefe",
            command=self.klasse_geaendert,
        )
        self.rb_ery.grid(row=4, column=0, padx=12, pady=4, sticky="w")
        self.rb_leu.grid(row=5, column=0, padx=12, pady=4, sticky="w")
        self.rb_hef.grid(row=6, column=0, padx=12, pady=4, sticky="w")

        # Anzeige: Zählerstand
        ctk.CTkLabel(
            self.panel_links, text="Zähler", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=7, column=0, padx=12, pady=(12, 4), sticky="w")

        self.lbl_zaehler = ctk.CTkLabel(
            self.panel_links,
            text="Ery: 0\nLeuko: 0\nHefe: 0\nGesamt: 0",
            justify="left",
        )
        self.lbl_zaehler.grid(row=8, column=0, padx=12, pady=(0, 10), sticky="w")

        # ===== Navigation (Autosave) =====
        frame_nav = ctk.CTkFrame(self.panel_links)
        frame_nav.grid(row=9, column=0, padx=12, pady=6, sticky="ew")
        frame_nav.grid_columnconfigure((0, 1), weight=1)

        # Button: "Vorher"
        self.btn_vor = ctk.CTkButton(
            frame_nav, text="← Vorher", command=self.vorheriges_bild
        )
        # Button: "Nächstes"
        self.btn_weiter = ctk.CTkButton(
            frame_nav, text="Nächstes →", command=self.naechstes_bild
        )
        self.btn_vor.grid(row=0, column=0, padx=(0, 6), pady=8, sticky="ew")
        self.btn_weiter.grid(row=0, column=1, padx=(6, 0), pady=8, sticky="ew")

        # Button: "Speichern"
        self.btn_speichern = ctk.CTkButton(
            self.panel_links, text="Speichern", command=self.speichern
        )
        self.btn_speichern.grid(row=10, column=0, padx=12, pady=(6, 4), sticky="ew")

        self.lbl_status = ctk.CTkLabel(
            self.panel_links, text="", wraplength=250, justify="left"
        )
        self.lbl_status.grid(row=11, column=0, padx=12, pady=(8, 12), sticky="w")

        # ===== Panel: rechts =====
        self.panel_rechts = ctk.CTkFrame(self)
        self.panel_rechts.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.panel_rechts.grid_columnconfigure(0, weight=1)
        self.panel_rechts.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.panel_rechts, bg="#1f1f1f", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.lbl_info = ctk.CTkLabel(self.panel_rechts, text="", anchor="w")
        self.lbl_info.place(relx=0.01, rely=0.01)

    # =========================
    # 6 ) Events: binden
    # =========================
    # Bindet Funktionen an events (user-input)
    def _events_binden(self):
        self.canvas.bind("<MouseWheel>", self.zoom_mausrad)

        self.canvas.bind("<ButtonPress-1>", self.maus_links_down)
        self.canvas.bind("<B1-Motion>", self.maus_links_move)
        self.canvas.bind("<ButtonRelease-1>", self.maus_links_up)

        self.canvas.bind("<Button-3>", self.rechtsklick_loeschen)
        self.canvas.bind("<Configure>", lambda e: self.neu_zeichnen())

    # =========================
    # 7 ) Funktionen: Bilder/Ordner und JSONs
    # =========================
    # ===== Ordner: auswählen =====
    def ordner_auswaehlen(self):
        start = (
            str(self.ordner_bilder) if self.ordner_bilder.exists() else str(Path.cwd())
        )
        ordner = filedialog.askdirectory(initialdir=start)
        if not ordner:
            return

        ordner = Path(ordner)
        bilder = sorted(
            [p for p in ordner.iterdir() if p.is_file() and ist_bilddatei(p)]
        )

        if not bilder:  # Hinweismeldung
            messagebox.showwarning(
                "Keine Bilder", "Im Ordner wurden keine Bilddateien gefunden."
            )
            return

        self.bilder_liste = bilder
        self.bild_index = 0
        self.lbl_ordner.configure(text=str(ordner))
        self.status(f"{len(self.bilder_liste)} Bilder geladen.")  # Statusmeldung
        self.bild_laden()

    # ===== Bild: laden =====
    def bild_laden(self):
        if not self.bilder_liste:
            return

        bild_pfad = self.bilder_liste[self.bild_index]

        try:
            self.bild_pil = Image.open(bild_pfad).convert("RGB")
        except Exception as ex:  # Fehlermeldung
            messagebox.showerror(
                "Fehler", f"Konnte Bild nicht laden:\n{bild_pfad}\n\n{ex}"
            )
            return

        # Punkte laden (falls JSON vorhanden)
        self.punkte = []
        rel_path = bild_pfad.relative_to(self.projekt_ordner / "data")
        label_ordner = self.ordner_labels_root / rel_path.parent
        label_pfad = label_ordner / f"{bild_pfad.stem}_points.json"
        if label_pfad.exists():
            try:
                daten = json.loads(label_pfad.read_text(encoding="utf-8"))
                for pt in daten.get("points", []):
                    if pt.get("class") in KLASSEN:
                        self.punkte.append(
                            {
                                "x": float(pt["x"]),
                                "y": float(pt["y"]),
                                "class": pt["class"],
                            }
                        )
            except Exception:  # falls JSONs fehlerhaft
                self.punkte = []

        # Ansicht neu zeichnen
        self.ansicht_fit()
        self.zaehler_aktualisieren()
        self.neu_zeichnen()
        self.info_aktualisieren()

    # =========================
    # 8 ) Navigation: Bilder
    # =========================
    # Vorher
    def vorheriges_bild(self):
        if not self.bilder_liste:
            return

        self.speichern()

        self.bild_index -= 1
        if self.bild_index < 0:
            self.bild_index = len(self.bilder_liste) - 1

        self.bild_laden()

    # Nächstes
    def naechstes_bild(self):
        if not self.bilder_liste:
            return

        self.speichern()

        self.bild_index += 1
        if self.bild_index >= len(self.bilder_liste):
            self.bild_index = 0

        self.bild_laden()

    # =========================
    # 9 ) Speichern (JSON)
    # =========================
    def speichern(self):
        if not self.bilder_liste:
            return

        # ===== JSONs passend zu Bilder speichern =====
        bild_pfad = self.bilder_liste[self.bild_index]
        rel_path = bild_pfad.relative_to(self.projekt_ordner / "data")
        label_ordner = self.ordner_labels_root / rel_path.parent
        label_pfad = label_ordner / f"{bild_pfad.stem}_points.json"

        daten = {"image": bild_pfad.name, "points": self.punkte}

        try:
            label_pfad.write_text(
                json.dumps(daten, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.status(f"Autosave: {label_pfad.name}")
        except Exception as ex:  # Fehlermeldung
            messagebox.showerror(
                "Fehler", f"Konnte nicht speichern:\n{label_pfad}\n\n{ex}"
            )

    # =========================
    # 10 ) Klasse ändern
    # =========================
    def klasse_geaendert(self):
        self.aktuelle_klasse = self.var_klasse.get()
        self.status(f"Klasse: {KLASSEN_TEXT[self.aktuelle_klasse]}")

    # =========================
    # 11 ) Koordinaten umrechnen
    # =========================
    # Canvas --> Bildkoordinaten
    def canvas_zu_bild(self, cx, cy):
        ix = (cx - self.verschiebung_x) / self.zoom
        iy = (cy - self.verschiebung_y) / self.zoom
        return ix, iy

    # Bild --> Canvas-Koordinaten
    def bild_zu_canvas(self, ix, iy):
        cx = ix * self.zoom + self.verschiebung_x
        cy = iy * self.zoom + self.verschiebung_y
        return cx, cy

    # =========================
    # 12 ) Event: Mausinput
    # =========================
    def ansicht_fit(self):
        if self.bild_pil is None:
            return

        # ===== Bild auf Canvas-Größe skalieren =====
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.bild_pil.size

        faktor = min(cw / iw, ch / ih) * 0.95
        self.zoom = begrenze(faktor, self.zoom_min, self.zoom_max)

        # ===== Bild zentrieren =====
        self.verschiebung_x = (cw - iw * self.zoom) / 2
        self.verschiebung_y = (ch - ih * self.zoom) / 2

    def zoom_mausrad(self, event):
        if self.bild_pil is None:
            return
        delta = event.delta / 120.0  # event.delta bei WIN oft 120/-120
        faktor = 1.1**delta
        self.zoom_um_maus(event.x, event.y, faktor)

    def zoom_um_maus(self, cx, cy, faktor):
        alt = self.zoom
        neu = begrenze(alt * faktor, self.zoom_min, self.zoom_max)
        if abs(neu - alt) < 1e-6:
            return

        ix, iy = self.canvas_zu_bild(cx, cy)
        self.zoom = neu
        self.verschiebung_x = cx - ix * self.zoom
        self.verschiebung_y = cy - iy * self.zoom

        self.neu_zeichnen()
        self.info_aktualisieren()

    def maus_links_down(self, event):
        self._maus_start = (event.x, event.y)  # Startpunkt (x,y)
        self._pan_start = (event.x, event.y)
        self._hat_gezogen = False

    def maus_links_move(self, event):
        if self._pan_start is None:
            return

        ax, ay = self._pan_start
        dx = event.x - ax
        dy = event.y - ay

        # ===== Ziehen oder Nichtziehen, ... =====
        if (
            abs(event.x - self._maus_start[0]) > DRAG_SCHWELLE
            or abs(event.y - self._maus_start[1]) > DRAG_SCHWELLE
        ):
            self._hat_gezogen = True

        self.verschiebung_x += dx
        self.verschiebung_y += dy
        self._pan_start = (event.x, event.y)

        self.neu_zeichnen()

    def maus_links_up(self, event):
        if self._maus_start is None:
            return

        if not self._hat_gezogen:
            self.punkt_setzen(event.x, event.y)

        self._maus_start = None
        self._pan_start = None
        self._hat_gezogen = False

    # =========================
    # 13 ) Punkte: setzen / löschen
    # =========================
    # Punkt setzen
    def punkt_setzen(self, cx, cy):
        if self.bild_pil is None:
            return

        iw, ih = self.bild_pil.size
        ix, iy = self.canvas_zu_bild(cx, cy)

        if ix < 0 or iy < 0 or ix >= iw or iy >= ih:  # Nur setzen, wenn im Bildbereich
            return

        self.punkte.append(
            {"x": round(ix, 3), "y": round(iy, 3), "class": self.aktuelle_klasse}
        )

        self.zaehler_aktualisieren()
        self.neu_zeichnen()
        self.info_aktualisieren()
        self.status(f"Punkt gesetzt: {KLASSEN_TEXT[self.aktuelle_klasse]}")

    # Punkt löschen
    def rechtsklick_loeschen(self, event):
        if not self.punkte:
            return

        # Nächstgelegenen Punkt suchen
        bester_index = -1
        bester_abstand2 = LOESCH_RADIUS_SCREEN**2

        for i, p in enumerate(self.punkte):
            cx, cy = self.bild_zu_canvas(p["x"], p["y"])
            d2 = (cx - event.x) ** 2 + (cy - event.y) ** 2
            if d2 <= bester_abstand2:
                bester_abstand2 = d2
                bester_index = i

        # Punkt löschen
        if bester_index >= 0:
            entfernt = self.punkte.pop(bester_index)
            self.zaehler_aktualisieren()
            self.neu_zeichnen()
            self.info_aktualisieren()
            self.status(f"Punkt gelöscht: {KLASSEN_TEXT[entfernt['class']]}")

    # =========================
    # 14 ) Anzeige: neu zeichnen
    # =========================
    # Neu zeichnen
    def neu_zeichnen(self):
        self.canvas.delete("all")
        if self.bild_pil is None:
            return

        iw, ih = self.bild_pil.size
        sw = int(iw * self.zoom)
        sh = int(ih * self.zoom)
        if sw <= 1 or sh <= 1:
            return

        # ===== Bild zeichnen =====
        bild_skaliert = self.bild_pil.resize((sw, sh), Image.BILINEAR)
        self.bild_tk = ImageTk.PhotoImage(bild_skaliert)

        self.canvas.create_image(
            self.verschiebung_x, self.verschiebung_y, image=self.bild_tk, anchor="nw"
        )

        # ===== Punkte zeichnen =====
        r = max(2, int(PUNKT_RADIUS))
        for p in self.punkte:
            cx, cy = self.bild_zu_canvas(p["x"], p["y"])
            farbe = FARBEN.get(p["class"], "#ffffff")
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r, outline=farbe, width=2
            )
            self.canvas.create_line(cx - r - 2, cy, cx + r + 2, cy, fill=farbe, width=2)
            self.canvas.create_line(cx, cy - r - 2, cx, cy + r + 2, fill=farbe, width=2)

    # =========================
    # 15 ) Status: aktualisieren
    # =========================
    # Zähler
    def zaehler_aktualisieren(self):
        zaehler = {"ery": 0, "leuko": 0, "hefe": 0}
        for p in self.punkte:
            if p["class"] in zaehler:
                zaehler[p["class"]] += 1

        gesamt = zaehler["ery"] + zaehler["leuko"] + zaehler["hefe"]
        self.lbl_zaehler.configure(
            text=f"Ery: {zaehler['ery']}\nLeuko: {zaehler['leuko']}\nHefe: {zaehler['hefe']}\nGesamt: {gesamt}"
        )

    # Infos
    def info_aktualisieren(self):
        if not self.bilder_liste or self.bild_pil is None:
            self.lbl_info.configure(text="")
            return

        pfad = self.bilder_liste[self.bild_index]
        iw, ih = self.bild_pil.size
        self.lbl_info.configure(
            text=f"sdgsbdfvssfgefx{self.bild_index + 1}/{len(self.bilder_liste)} – {pfad.name} – {iw}x{ih} – Zoom {self.zoom:.2f}x"
        )

    # Status
    def status(self, text):
        self.lbl_status.configure(text=text)


# =========================
# 16 ) Start der Anwendung
# =========================
if __name__ == "__main__":
    app = LabelMaker()
    app.mainloop()
