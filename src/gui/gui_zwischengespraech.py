"""
Modul: aust_gui.py
Beschreibung:
Dieses Skript ist die grafische Benutzeroberflächedes Projekts Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse von mikroskopischen Zellstrukturen.

Autor: Max Sielhorst
Co-Autor: Marlon Aust für gesamten Hell-/Dunkelmodus, Integrierung KI und Validierungsfunktion
Co-Autor: Sven Klapp für Error-Handling
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# ============================
# 1 ) Bibliotheken importieren
# ============================
import threading
import customtkinter as ctk  # Modernes "Tkinter" mit Darkmode + besseren Widgets
import tkinter as tk  # Standard Tkinter (hier genutzt für tk.Frame / tk.Label)
from tkinter import (filedialog, messagebox)  # Datei-Dialoge (open/save) und Popup-Meldungen
import cv2  # OpenCV: Kamera-Input, Frame lesen, Bild speichern, Farbkonvertierung
import os  # Dateisystem: Ordner erstellen, Pfade bauen, Dateien prüfen/listen
import sys  # Pfad-Manipulation für KI-Module: Ordner erstellen, Pfade bauen, Dateien prüfen/listen
import shutil  # Dateikopien (copy2) für "Export all"
import numpy as np  # Bilddaten als Arrays (RGB/BGR) bearbeiten
import time
import re  # Text-Updates (Analyse-Label)
from datetime import datetime  # Zeitstempel für Dateinamen
from pathlib import Path  # robuste Pfade (KI-Model/Referenz)  # Zeitstempel für Dateinamen
from PIL import (Image, ImageTk,)  # Pillow: Bilder laden/konvertieren + Tk-kompatible Anzeige

# ============================
#  2 ) Globale Einstellungen
# ============================
appearance = "dark"  # Standard-Farbmodus
colormode = ("Hellmodus")  # Farbwechsel-Button Text
ctk.set_appearance_mode(appearance)  # globales Theme setzen
ctk.set_default_color_theme("blue")  # Standardfarbtheme

deltatol = 0.01  # Validation: Toleranz (dezimal)


# ============================
# 3 ) Hauptanwendung
# ============================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()  # initialisiert das CTk-Hauptfenster

        # ===== Fenster Einstellungen =====
        self.title("visCell")  # Fenstertitel oben
        self.geometry("1280x720")  # Startgröße
        self.minsize(1100, 700)  # Mindestgröße (User kann nicht kleiner ziehen)

        # ===== Validierung =====
        # Sollwert (Referenz) – wird vorerst vordefiniert genutzt
        self.soll_ery = 113  # Referenzwert (Soll)
        self.soll_leuko = 2 
        self.soll_hefe = 0

        # KI-Ausgaben (werden bei "Analyse starten" gesetzt)
        self.ki_ery = 0
        self.ki_leuko = 0
        self.ki_hefe = 0

        self.validierung_ok = False  # Validierung ok/nok

        # ===== KI: Pfade / Defaults =====
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_path = self.project_root / "runs" / "viscell_unet_512_cpu_classfix" / "model_best.keras"
        # Referenzbilder werden später hier abgelegt
        self.ref_dir = self.project_root / "viscell" / "data" / "ref"

        # Analyse-Thread-Flag
        self._analysis_running = False

        # ===== Kamera / Video Status-Variablen =====
        self.cap = None  # OpenCV VideoCapture Objekt (None = keine Kamera geöffnet)
        self.is_streaming = False  # Flag: True = Live-Loop soll laufen, False = kein Loop
        self.current_cam_index = None  # gemerkter Index der aktiven Kamera (0/1/2/...)
        self.last_frame = None  # letzter Frame als BGR (OpenCV Standardformat)
        self.current_image_path = None  # Pfad des aktuell importierten/angezeigten Bildes (falls vorhanden)

        # ===== Tk-Image Referenz + after()-ID =====
        self._tk_img = None  # Referenz auf PhotoImage (wichtig: sonst zeigt Tk das Bild nicht)
        self._after_id = None  # ID des geplanten self.after(...) Calls (zum sauberen Abbrechen)

        # Dropdown Variable (Kamera Auswahl)
        self.camera_var = ctk.StringVar(value="Kamera auswählen...")  # Starttext im Dropdown

        # Freeze Checkbox Variable
        self.freeze_after_capture_var = ctk.BooleanVar(value=False)  # False = normal, True = Freeze nach Snapshot

        # ===== Speicherordner für Snapshots/Captures =====
        self.capture_dir = "captures"  # Ordnername
        os.makedirs(self.capture_dir, exist_ok=True)  # Ordner erstellen falls nicht vorhanden

        # ===== GUI Widget Referenzen (werden später in _build_gui gesetzt) =====
        self.live_view = None
        self.video_container = None
        self.video_label = None
        self.video_text = None

        self.analysis_frame = None
        self.analysis_label = None
        self.cam_status_label = None
        self.darkmode_btn = None

        # Button-Referenz "Bild aufnehmen"
        self.capture_btn = None
        # Button-Referenz "Analyse starten"
        self.btn_analyze = None

        # ===== self.farbwechsel =====
        self.appearance = appearance
        self.colormode_text = colormode

        # ===== Farben für Anzeigezustände =====
        self.placeholder_bg = "#BDBDBD"  # hellgrau: wenn kein Live-Feed/Bild
        self.live_bg = "black"  # schwarz: wenn Live-Feed läuft / Bild angezeigt wird

        # =========================
        # Status-State (damit Themewechsel NICHT deine Validierungsfarben überschreibt)
        # =========================
        self._status_kind = "info"     # "info" | "warning" | "ok" | "error" | "custom"
        self._status_text = "Bereit"   # letzter Text
        self._status_color = None      # letzte (oder berechnete) Farbe

        # Fenster-Schließen abfangen (damit Kamera sauber freigegeben wird)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ===== GUI bauen =====
        self._build_gui()
        self._build_footer()

        # Dropdown trace: wenn camera_var geändert wird -> Kamerawechsel ausführen
        self.camera_var.trace_add("write", self._camera_var_changed)

        # Initial: Placeholder anzeigen
        self._clear_video_label()

        # Initial: Button-Status setzen (disabled, solange keine Kamera aktiv)
        self._update_capture_button_state()

        # Initial: Status sauber setzen (passend zum Theme)
        self._apply_status_style_on_theme_change()

    # =========================
    # 3.1 ) Helper: "Bild aufnehmen" Button aktiv/deaktiv setzen
    # =========================
    def _update_capture_button_state(self):
        """
        Idee:
        - Wenn KEINE Kamera ausgewählt ist (Dropdown steht auf "Kamera auswählen..." oder "Keine Kamera gefunden"),
          dann soll "Bild aufnehmen" ausgegraut und nicht anklickbar sein.
        - Wenn eine echte Kamera gewählt ist ("Camera X"), dann ist der Button aktiv.
        """
        choice = self.camera_var.get()

        enable = False
        if isinstance(choice, str) and choice.startswith("Camera "):
            enable = True

        if choice in ("Kamera auswählen...", "Keine Kamera gefunden"):
            enable = False

        if self.capture_btn is not None:
            self.capture_btn.configure(state="normal" if enable else "disabled")

    # =========================
    # 3.2 ) Status-Farben je nach Theme (gut lesbar)
    # =========================
    def _status_color_for_kind(self, kind: str) -> str:
        """
        Liefert eine gut lesbare Status-Farbe abhängig vom Theme.
        (Helles Orange im Darkmode, dunkles Orange im Lightmode, usw.)
        """
        is_dark = (appearance == "dark")

        if kind == "ok":
            return "#00FF66" if is_dark else "#005823"
        if kind == "warning":
            # helles Orange in Darkmode / dunkleres Orange in Lightmode
            return "#FFAA00" if is_dark else "#A86D00"
        if kind == "error":
            return "#FF4444" if is_dark else "#DB0101"
        # "info" / default
        return "#00B7FF" if is_dark else "#000000"

    # =========================
    # 3.3 ) Themewechsel -> Status-Farbe neu anwenden (ohne Überschreiben!)
    # =========================
    def _apply_status_style_on_theme_change(self):
        """
        Wichtig:
        - Beim Themewechsel darf NICHT pauschal text_color gesetzt werden (das killt Validierungsfarben).
        - Stattdessen wird die aktuelle Status-Art (kind) neu berechnet und wieder angewendet.
        """
        if self._status_kind != "custom":
            self._status_color = self._status_color_for_kind(self._status_kind)

        # Status neu zeichnen (Text bleibt, Farbe wird angepasst)
        self._status(self._status_text, kind=self._status_kind if self._status_kind != "custom" else None)

    # =========================
    # 3.4 ) Status setzen (einheitlich, speichert State)
    # =========================
    def _status(self, text: str, farbe: str | None = None, kind: str | None = None):
        """
        Einheitliche Status-Funktion:
        - Wenn kind gesetzt ist, wird die Farbe automatisch passend zum Theme gewählt.
        - Wenn farbe gesetzt ist (ohne kind), wird eine Custom-Farbe gespeichert.
        """
        if kind is not None:
            self._status_kind = kind
            self._status_color = self._status_color_for_kind(kind)
        else:
            if farbe is not None:
                self._status_kind = "custom"
                self._status_color = farbe

        self._status_text = text

        def gui_update():
            if self.status_label is not None:
                self.status_label.configure(text=self._status_text, text_color=self._status_color or "#00B7FF")
                # optional: Status nach oben holen (falls Layer-Probleme)
                try:
                    self.status_label.lift()
                except Exception:
                    pass

        self.after(0, gui_update)

    # ===== Helper: Background für Live/Placeholder setzen =====
    def _set_live_background(self, color: str):
        # CTkFrame Hintergrund setzen
        if self.live_view is not None:
            self.live_view.configure(fg_color=color)

        # Tk Widgets Hintergrund setzen (weil tk.Frame / tk.Label separate bg-Property haben)
        if self.video_container is not None:
            self.video_container.configure(bg=color)
        if self.video_label is not None:
            self.video_label.configure(bg=color)

        # WICHTIG:
        # Hier darf NICHT die Status-Farbe gesetzt werden!
        # (Sonst überschreibst du z.B. Orange/Rot nach der Validierung.)

    # ===== Farbwechsel =====
    def _switchcolor(self):
        global appearance
        global colormode

        if appearance == "light":
            appearance = "dark"
            colormode = "Hellmodus"
        else:
            appearance = "light"
            colormode = "Dunkelmodus"

        # ===== GUI anpassen =====
        ctk.set_appearance_mode(appearance)

        self.appearance = appearance
        self.colormode_text = colormode

        if self.darkmode_btn is not None:
            self.darkmode_btn.configure(text=colormode)

        if self.is_streaming:
            self._set_live_background(self.live_bg)
        else:
            self._set_live_background(self.placeholder_bg)

        # NEU: Status-Farbe passend zum Theme neu anwenden (damit Orange immer lesbar bleibt)
        self._apply_status_style_on_theme_change()

    # =========================
    # 5 ) Footer (unten rechts: Theme Toggle Button)
    # =========================
    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=30)  # Footer-Leiste
        footer.pack(fill="x", side="bottom")

        # ===== Button: Farbwechsel =====
        self.darkmode_btn = ctk.CTkButton(
            footer,
            text=self.colormode_text,
            command=self._switchcolor,
            width=110
        )
        self.darkmode_btn.pack(side="right", padx=5, pady=5)

    # ===== Kameras finden (0..max_index) =====
    def _detect_cameras(self, max_index: int = 5):
        found = []  # Liste der gefundenen Kameras (Strings)

        # Kamera-Indices testen (z.B. 0..5)
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = Windows DirectShow (oft zuverlässiger)
            if cap.isOpened():  # prüfen ob geöffnet
                ret, _ = cap.read()  # Testframe lesen
                if ret:  # wenn erfolgreich -> Kamera gilt als vorhanden
                    found.append(f"Camera {i}")
            cap.release()  # wichtig: testweise Kamera wieder schließen

        return found

    # =========================
    # 6 ) Kamera Dropdown via trace
    # =========================
    def _camera_var_changed(self, *_):
        choice = self.camera_var.get()  # aktuellen Auswahltext holen
        self._on_camera_selected(choice)  # Auswahl behandeln

        # Nach jeder Dropdown-Änderung den Button-Status aktualisieren
        self._update_capture_button_state()

    # Falls "Camera X" gewählt -> Kamera starten
    def _on_camera_selected(self, choice: str):
        if isinstance(choice, str) and choice.startswith("Camera "):
            idx = int(choice.split()[-1])  # Index aus "Camera 0" etc. extrahieren
            self._start_camera(idx)  # Kamera starten
        else:
            # Falls keine echte Kamera gewählt -> stoppen + Placeholder
            self._stop_camera()
            self._clear_video_label()

    # ===== Freeze Toggle =====
    def _on_freeze_toggle(self):
        # Wenn Freeze AUSgeschaltet wird und eine Kamera offen ist, soll Live wieder laufen
        if not self.freeze_after_capture_var.get():  # Freeze = False
            if (self.cap is not None and self.current_cam_index is not None and not self.is_streaming):
                self.is_streaming = True
                self._set_status("Live-Feed fortgesetzt (Freeze aus)")
                self._hide_overlay_text()
                self._set_live_background(self.live_bg)
                self._update_frame_loop()

    # ===== Overlay Handling (Text über dem Video) =====
    def _hide_overlay_text(self):
        try:
            self.video_text.place_forget()
        except Exception:
            pass

    # ===== Overlay-Text setzen + zentriert anzeigen =====
    def _show_overlay_text(self, text: str):
        self.video_text.configure(text=text, text_color=("black", "black"))
        self.video_text.place(relx=0.5, rely=0.5, anchor="center")
        self.video_text.lift()

    # =====Placeholder Hintergrund aktivieren =====
    def _clear_video_label(self):
        self._set_live_background(self.placeholder_bg)

        self._tk_img = None
        if self.video_label is not None:
            self.video_label.configure(image="")
            self.video_label.image = None

        if self.video_text is not None:
            self._show_overlay_text("Kein Bild vorhanden\n(bitte Kamera auswählen)")

        self.update_idletasks()

    # ===== Kamera Start/Stop =====
    def _start_camera(self, index: int):
        self._stop_camera()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            self._set_live_background(self.placeholder_bg)
            self._show_overlay_text(f"Kamera {index} konnte nicht geöffnet werden")
            self._update_capture_button_state()
            return

        self.current_cam_index = index
        self.is_streaming = True

        self._set_live_background(self.live_bg)
        self._hide_overlay_text()
        self._update_frame_loop()

        self._update_capture_button_state()

    # ===== Streaming deaktivieren =====
    def _stop_camera(self):
        self.is_streaming = False
        self.current_cam_index = None

        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self._tk_img = None
        self._set_live_background(self.placeholder_bg)
        self._update_capture_button_state()

    # ===== Resize helper: FILL (Panel füllen, ggf. Crop) =====
    def _resize_fill(self, rgb, target_w, target_h):
        h, w = rgb.shape[:2]
        if w <= 0 or h <= 0:
            return rgb

        scale = max(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x0 = (new_w - target_w) // 2
        y0 = (new_h - target_h) // 2

        return resized[y0: y0 + target_h, x0: x0 + target_w]

    # ===== Helper: RGB Bild im Panel anzeigen =====
    def _display_rgb_in_video_label(self, rgb: np.ndarray):
        if self.video_label is None:
            return

        self._set_live_background(self.live_bg)
        self._hide_overlay_text()

        self.video_label.update_idletasks()
        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()

        if target_w <= 50 or target_h <= 50:
            target_w, target_h = 1280, 720

        rgb = self._resize_fill(rgb, target_w, target_h)

        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        self._tk_img = ImageTk.PhotoImage(pil_img)
        self.video_label.configure(image=self._tk_img)
        self.video_label.image = self._tk_img

    # =========================
    # 7 ) Live Loop (liest Frames und zeigt sie)
    # =========================
    def _update_frame_loop(self):
        if not self.is_streaming or self.cap is None:
            return

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.last_frame = frame
                self.current_image_path = None
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.video_label.update_idletasks()
                target_w = self.video_label.winfo_width()
                target_h = self.video_label.winfo_height()

                if target_w > 50 and target_h > 50:
                    rgb = self._resize_fill(rgb, target_w, target_h)
                    pil_img = Image.fromarray(rgb)
                    pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
                else:
                    pil_img = Image.fromarray(rgb)

                self._tk_img = ImageTk.PhotoImage(pil_img)
                self.video_label.configure(image=self._tk_img)
                self.video_label.image = self._tk_img

        except Exception as e:
            print("Fehler im Live-Loop:", e)

        self._after_id = self.after(33, self._update_frame_loop)

    # ===== Snapshot (speichert in captures/) =====
    def _save_snapshot(self):
        if self.last_frame is None:
            self._set_status("Kein Frame vorhanden (Kamera läuft? / Bild importiert?)")
            return

        ts = datetime.now().strftime("%d%m%Y_%H%M%S_%f")[:-3]
        cam = (f"cam{self.current_cam_index}" if self.current_cam_index is not None else "img")
        filename = f"snapshot_{cam}_{ts}.png"

        path = os.path.join(self.capture_dir, filename)
        ok = cv2.imwrite(path, self.last_frame)

        if not ok:
            self._set_status("Speichern fehlgeschlagen")
            return

        if self.freeze_after_capture_var.get() and self.cap is not None:
            self.is_streaming = False

            if self._after_id is not None:
                try:
                    self.after_cancel(self._after_id)
                except Exception:
                    pass
                self._after_id = None

            self.current_image_path = path

            try:
                pil_img = Image.open(path).convert("RGB")
                rgb = np.array(pil_img)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.last_frame = bgr

                self._display_rgb_in_video_label(rgb)
                self._set_status(f"Gespeichert & geöffnet (Freeze): {path}")

            except Exception as e:
                rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                self._display_rgb_in_video_label(rgb)
                self._set_status(f"Gespeichert (Freeze) - Laden fehlgeschlagen: {e}")
        else:
            self._set_status(f"Gespeichert: {path}")

    # =========================
    # 8 ) Bild importieren (Dialog startet in captures/)
    # =========================
    def _import_image(self):
        self._stop_camera()

        file_path = filedialog.askopenfilename(
            title="Bild importieren",
            initialdir=self.capture_dir,
            filetypes=[
                ("Bilddateien", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Alle Dateien", "*.*"),
            ],
        )

        if not file_path:
            self._set_status("Import abgebrochen")
            return

        try:
            pil_img = Image.open(file_path).convert("RGB")
            rgb = np.array(pil_img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            self.last_frame = bgr
            self.current_image_path = file_path
            self._display_rgb_in_video_label(rgb)
            self._set_status(f"Importiert: {os.path.basename(file_path)}")

        except Exception as e:
            self._set_status("Import fehlgeschlagen")
            messagebox.showerror("Fehler", f"Bild konnte nicht importiert werden:\n{e}")

    # =========================
    # 9 ) Bild exportieren (Save-Dialog; startet in captures/)
    # =========================
    def _export_image(self):
        if self.last_frame is None:
            self._set_status("Kein Bild zum Exportieren vorhanden")
            return

        ts = datetime.now().strftime("%d%m%Y_%H%M%S")
        default_name = f"export_{ts}.png"

        save_path = filedialog.asksaveasfilename(
            title="Bild exportieren",
            initialdir=self.capture_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif *.tiff"),
            ],
        )

        if not save_path:
            self._set_status("Export abgebrochen")
            return

        try:
            ok = cv2.imwrite(save_path, self.last_frame)

            if ok:
                self._set_status(f"Exportiert: {save_path}")
            else:
                self._set_status("Export fehlgeschlagen")
                messagebox.showerror("Fehler", "Export fehlgeschlagen (cv2.imwrite gab False zurück).")

        except Exception as e:
            self._set_status("Export fehlgeschlagen")
            messagebox.showerror("Fehler", f"Bild konnte nicht exportiert werden:\n{e}")

    # ===== Optional: Alles aus captures/ exportieren (kopieren) =====
    def _export_all_captures(self):
        target_dir = filedialog.askdirectory(
            title="Zielordner wählen (alle Captures exportieren)",
            initialdir=os.path.abspath(self.capture_dir)
        )

        if not target_dir:
            self._set_status("Export abgebrochen")
            return

        try:
            count = 0
            for name in os.listdir(self.capture_dir):
                src = os.path.join(self.capture_dir, name)

                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(target_dir, name))
                    count += 1

            self._set_status(f"Exportiert: {count} Datei(en) nach {target_dir}")

        except Exception as e:
            self._set_status("Export aller Captures fehlgeschlagen")
            messagebox.showerror("Fehler", f"Konnte Captures nicht exportieren:\n{e}")

    # ===== Status Text in GUI setzen (Kamera-Status links oben) =====
    def _set_status(self, msg: str):
        if self.cam_status_label is not None:
            self.cam_status_label.configure(text=msg)

    # =========================
    # 10 ) Metadaten Popup (Formular)
    # =========================
    def _open_metadata_popup(self, analysis_label: ctk.CTkLabel):
        popup = ctk.CTkToplevel(self)
        popup.title("Metadaten bearbeiten")
        popup.geometry("400x400")
        popup.grab_set()

        title_label = ctk.CTkLabel(
            popup,
            text="Metadaten bearbeiten",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title_label.pack(pady=10)

        fields = [
            "Analysedatum",
            "Prüfer",
            "Labor",
            "Equipment",
            "Vergrößerung",
            "Probennummer",
            "Notizen",
        ]

        entries = {}

        for field in fields:
            frame = ctk.CTkFrame(popup)
            frame.pack(fill="x", padx=20, pady=5)

            label = ctk.CTkLabel(frame, text=field, width=100, anchor="w")
            label.pack(side="left")

            entry = ctk.CTkEntry(frame)
            entry.pack(side="right", fill="x", expand=True)

            entries[field] = entry

        def save_metadata():
            analysedatum = entries["Analysedatum"].get()
            pruefer = entries["Prüfer"].get()
            labor = entries["Labor"].get()
            equipment = entries["Equipment"].get()
            vergroesserung = entries["Vergrößerung"].get()
            probennummer = entries["Probennummer"].get()
            notizen = entries["Notizen"].get()

            analysis_label.configure(
                text=(
                    "ANALYSE_ERGEBNIS\n\n"
                    f"Datum: {analysedatum}\n"
                    f"Prüfer: {pruefer}\n"
                    f"Labor: {labor}\n\n"
                    f"Mikroskop: {equipment}\n"
                    f"Vergrößerung: {vergroesserung}\n"
                    f"Probe Nummer: {probennummer}\n\n"
                    "Erythrozyten Anzahl:      \n"
                    "Leukozyten Anzahl:     \n"
                    "Hefezellen Anzahl:    \n\n"
                    f"Notizen: {notizen}"
                )
            )
            popup.destroy()

        save_btn = ctk.CTkButton(popup, text="Speichern", command=save_metadata)
        save_btn.pack(pady=20)

    
    # =========================
    # 10.5 ) Analyse (KI)
    # =========================
    def start_analysis(self):
        """Startet die KI-Analyse für das aktuell im View-Fenster angezeigte Bild."""
        if self._analysis_running:
            return

        if self.last_frame is None:
            self._status("Kein Bild vorhanden – erst Kamera wählen / Bild importieren", kind="warning")
            return

        if self.btn_analyze is not None:
            self.btn_analyze.configure(state="disabled")

        self._analysis_running = True
        self._show_progress()

        thread = threading.Thread(target=self._analysis_process, daemon=True)
        thread.start()

    def _analysis_process(self):
        try:
            # Schritt 1: Modell prüfen
            self._status("Prüfe KI-Modell...", kind="info")
            self._update_progress(1, 5)

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model nicht gefunden: {self.model_path}")

            # Schritt 2: Bild für Inferenz bereitstellen (aktuelles View-Bild)
            self._status("Bereite Bild vor...", kind="info")
            self._update_progress(2, 5)

            # last_frame ist BGR (OpenCV) – wir schreiben temporär eine PNG-Datei
            tmp_dir = Path(self.capture_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / "_tmp_analysis.png"
            cv2.imwrite(str(tmp_path), self.last_frame)

            # Schritt 3: KI-Inferenz
            self._status("KI analysiert Bild...", kind="info")
            self._update_progress(3, 5)

            # Import hier, damit GUI schneller startet (TF lädt erst bei Bedarf)
            ai_src = (self.project_root / "src" / "ai").resolve()
            if str(ai_src) not in sys.path:
                sys.path.insert(0, str(ai_src))

            import numpy as _np  # noqa
            import infer_count_clean_v6 as infermod  # noqa

            class_weights = _np.array([1.0, 0.3, 1.2], dtype=_np.float32)

            counts = infermod.infer_and_count(
                model_path=Path(self.model_path),
                image_path=Path(tmp_path),
                spec=infermod.TileSpec(tile=512, overlap=64),
                quantile=0.9943,
                abs_thresh=0.0,
                max_fg=0.01,
                detect_dist=6,
                merge_dist=14,
                max_area=120,
                peak_rel=1.0,
                class_weights=class_weights,
                class_margin=0.02,
                ambiguous_policy="ery",
                leuko_min_abs=0.06,
                leuko_min_rel=0.55,
                hefe_min_abs=0.08,
                hefe_min_rel=0.60,
                leuko_margin_over_ery=0.015,
                debug=False,
            )

            # Schritt 4: Ergebnis übernehmen (Variablennamen aus GUI beibehalten)
            self._status("Übernehme Ergebnis...", kind="info")
            self._update_progress(4, 5)

            self.ki_ery = int(counts.get("ery", 0))
            self.ki_leuko = int(counts.get("leuko", 0))
            self.ki_hefe = int(counts.get("hefe", 0))

            # Schritt 5: GUI aktualisieren
            self._status("Analyse abgeschlossen ✅", kind="ok")
            self._update_progress(5, 5)

            self.after(0, lambda: self._update_analysis_counts(self.ki_ery, self.ki_leuko, self.ki_hefe))

        except Exception as e:
            self._status("Analyse fehlgeschlagen ❌", kind="error")
            self.after(0, lambda: messagebox.showerror("Analyse", f"Analyse fehlgeschlagen:\n{e}"))

        finally:
            self._analysis_running = False
            if self.btn_analyze is not None:
                self.after(0, lambda: self.btn_analyze.configure(state="normal"))

    def _update_analysis_counts(self, ery: int, leuko: int, hefe: int):
        """Aktualisiert die drei Ergebniszeilen im rechten Analysefenster."""
        if self.analysis_label is None:
            return

        txt = self.analysis_label.cget("text") or ""

        def repl(label: str, value: int, s: str) -> str:
            # Erlaubt sowohl '... Anzahl:' als auch '... Anzahl:   ' etc.
            pattern = rf"({re.escape(label)}\s*)(.*)"
            lines = s.splitlines()
            out = []
            replaced = False
            for line in lines:
                m = re.match(pattern, line)
                if m:
                    out.append(f"{m.group(1)}{value}")
                    replaced = True
                else:
                    out.append(line)
            if not replaced:
                out.append(f"{label} {value}")
            return "\n".join(out)

        txt = repl("Erythrozyten Anzahl:", ery, txt)
        txt = repl("Leukozyten Anzahl:", leuko, txt)
        txt = repl("Hefezellen Anzahl:", hefe, txt)

        self.analysis_label.configure(text=txt)

# =========================
    # 11 ) Validierung
    # =========================
    def start_validation(self):
        self.btn_validate.configure(state="disabled")
        self._show_progress()
        thread = threading.Thread(target=self._validation_process, daemon=True)
        thread.start()

    def _validation_process(self):
        time.sleep(0.4)

        # Schritte: Status (info)
        self._status("Lade Referenzdaten...", kind="info")
        self._update_progress(1, 5)
        time.sleep(0.1)

        soll = int(self.soll_ery)

        self._status("KI analysiert Referenzbild...", kind="info")
        self._update_progress(2, 5)
        time.sleep(0.8)

        ist = int(self.ki_ery)

        self._status("Prüfe Ergebnis (Regel A)...", kind="info")
        self._update_progress(3, 5)
        time.sleep(0.05)

        ok, tol = self._regel_check(ist=ist, soll=soll)
        self.validierung_ok = bool(ok)

        self._status("Ergebnis bereit (Bediener bestätigen)...", kind="info")
        self._update_progress(4, 5)
        time.sleep(0.6)

        self.after(0, lambda: self._popup_confirm(ist, soll, tol))
        self._update_progress(5, 5)

    def _regel_check(self, ist: int, soll: int) -> tuple[bool, int]:
        toleranz = max(1, round(soll * deltatol))
        untergrenze = soll - toleranz
        obergrenze = soll + toleranz

        ok = untergrenze <= ist <= obergrenze
        return ok, toleranz

    def _popup_confirm(self, ist: int, soll: int, tol: int):
        if self.validierung_ok:
            text = (
                f"Validierung erfolgreich.\n\n"
                f"Referenzwert: {soll}\n"
                f"Istwert: {ist}\n"
                f"Toleranz: ±{tol}%\n\n"
                f"Bestätigen Sie die Validierung?"
            )
        else:
            text = (
                f"Validierung nicht erfolgreich.\n\n"
                f"Referenzwert: {soll}\n"
                f"Istwert: {ist}\n"
                f"Toleranz: ±{tol}%\n\n"
                f"Trotzdem als erfolgreich bestätigen?"
            )

        confirmed = messagebox.askyesno("Validierung", text)

        # Ergebnis: Status (ok / warning / error) -> Farbe automatisch je Theme
        if confirmed and self.validierung_ok:
            self._status("Validierung bestätigt ✅", kind="ok")
        elif confirmed and not self.validierung_ok:
            self._status("Bestätigt, aber NICHT bestanden ⚠️", kind="warning")
        else:
            self._status("Nicht bestätigt ❌", kind="error")

        self.btn_validate.configure(state="normal")

    # ===== Progress GUI =====
    def _update_progress(self, count: int, total: int):
        total = max(1, int(total))
        percent = max(0, min(count / total, 1))

        def gui_update():
            self.progress.set(percent)
            self.progress_text.configure(text=f"{int(percent * 100)} %")

        self.after(0, gui_update)

    def _show_progress(self):
        self.progress.set(0)
        self.progress_text.configure(text="0 %")
        self.progress_frame.pack(pady=6)

    # =========================
    # 12 ) GUI bauen (Layout + Widgets)
    # =========================
    def _build_gui(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        main_frame.grid_columnconfigure(0, weight=6)
        main_frame.grid_columnconfigure(1, weight=0, minsize=360)
        main_frame.grid_rowconfigure(1, weight=1)

        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))

        detected = self._detect_cameras(max_index=5)
        camera_values = ["Kamera auswählen..."] + detected
        if len(detected) == 0:
            camera_values = ["Kamera auswählen...", "Keine Kamera gefunden"]

        self.camera_select = ctk.CTkOptionMenu(
            controls_frame,
            values=camera_values,
            variable=self.camera_var
        )
        self.camera_select.grid(row=0, column=0, padx=5)

        self.freeze_checkbox = ctk.CTkCheckBox(
            controls_frame,
            text="Nach Aufnahme pausieren",
            variable=self.freeze_after_capture_var,
            command=self._on_freeze_toggle
        )
        self.freeze_checkbox.grid(row=0, column=1, padx=10)

        self.btn_analyze = ctk.CTkButton(controls_frame, text="Analyse starten", command=self.start_analysis)
        self.btn_analyze.grid(row=0, column=2, padx=5)

        self.cam_status_label = ctk.CTkLabel(controls_frame, text="", width=400, anchor="w")
        self.cam_status_label.grid(row=0, column=3, padx=10)

        live_view = ctk.CTkFrame(main_frame, fg_color=self.placeholder_bg, corner_radius=0)
        live_view.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        live_view.grid_rowconfigure(0, weight=1)
        live_view.grid_columnconfigure(0, weight=1)
        self.live_view = live_view

        self.video_container = tk.Frame(live_view, bg=self.placeholder_bg, bd=0, highlightthickness=0)
        self.video_container.grid(row=0, column=0, sticky="nsew")
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(
            self.video_container,
            bg=self.placeholder_bg,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.video_text = ctk.CTkLabel(
            live_view,
            text="Kein Bild vorhanden\n(bitte Kamera auswählen)",
            font=ctk.CTkFont(size=18),
            fg_color="transparent",
        )
        self.video_text.place(relx=0.5, rely=0.5, anchor="center")

        bottom_controls = ctk.CTkFrame(main_frame)
        bottom_controls.grid(row=2, column=0, sticky="w", pady=10)

        ctk.CTkButton(bottom_controls, text="Bild importieren", command=self._import_image).grid(row=0, column=0, padx=5)
        ctk.CTkButton(bottom_controls, text="Bild exportieren", command=self._export_image).grid(row=0, column=1, padx=5)

        # Button: "Bild aufnehmen"
        self.capture_btn = ctk.CTkButton(bottom_controls, text="Bild aufnehmen", command=self._save_snapshot)
        self.capture_btn.grid(row=0, column=2, padx=5)

        ctk.CTkButton(bottom_controls, text="Alle Captures exportieren", command=self._export_all_captures).grid(row=0, column=3, padx=5)

        # Direkt nach dem Erstellen einmal den Zustand setzen
        self._update_capture_button_state()

        right_panel = ctk.CTkFrame(main_frame, width=360)
        right_panel.grid(row=0, column=1, rowspan=3, sticky="nsew")
        right_panel.grid_propagate(False)
        right_panel.grid_rowconfigure(1, weight=1)

        analysis_frame = ctk.CTkFrame(right_panel, fg_color="#BDBDBD", corner_radius=8, height=200)
        analysis_frame.pack(fill="x", padx=10, pady=10)
        self.analysis_frame = analysis_frame

        analysis_label = ctk.CTkLabel(
            analysis_frame,
            text=(
                "ANALYSE_ERGEBNIS\n\n"
                "Datum:  \nPrüfer:  \nLabor: \n\n"
                "Mikroskop:   \nVergrößerung:   \nProbe Nummer:   \n\n"
                "Erythrozyten Anzahl:   \nLeukozyten Anzahl:   \nHefezellen Anzahl: "
            ),
            justify="left",
            text_color="black"
        )
        analysis_label.pack(side="left", padx=10, pady=10)
        self.analysis_label = analysis_label

        export_frame = ctk.CTkFrame(right_panel, fg_color="transparent", height=200)
        export_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(
            export_frame,
            text="Metadaten ändern",
            command=lambda: self._open_metadata_popup(analysis_label),
        ).pack(side="left", pady=10)

        ctk.CTkButton(export_frame, text="Bericht exportieren").pack(side="left", padx=10, pady=10)

        slider_frame = ctk.CTkFrame(right_panel)
        slider_frame.pack(fill="x", padx=10, pady=10)

        row = ctk.CTkFrame(slider_frame, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(row, text="Sättigung").pack(side="left")
        value_label = ctk.CTkLabel(row, text="50%")
        value_label.pack(side="right")

        saturation_slider = ctk.CTkSlider(slider_frame, from_=0, to=100)
        saturation_slider.pack(fill="x", padx=10, pady=(0, 10))

        def on_sat(v):
            value_label.configure(text=f"{int(v)}%")

        saturation_slider.configure(command=on_sat)
        saturation_slider.set(50)
        on_sat(50)

        row_brightness = ctk.CTkFrame(slider_frame, fg_color="transparent")
        row_brightness.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(row_brightness, text="Helligkeit").pack(side="left")
        brightness_value_label = ctk.CTkLabel(row_brightness, text="50%")
        brightness_value_label.pack(side="right")

        brightness_slider = ctk.CTkSlider(slider_frame, from_=0, to=100)
        brightness_slider.pack(fill="x", padx=10, pady=(0, 10))

        def on_brightness(v):
            brightness_value_label.configure(text=f"{int(v)}%")

        brightness_slider.configure(command=on_brightness)
        brightness_slider.set(50)
        on_brightness(50)

        function_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        function_frame.pack(fill="x", padx=10, pady=10)

        button_row = ctk.CTkFrame(function_frame, fg_color="transparent")
        button_row.pack(fill="x")

        self.btn_validate = ctk.CTkButton(button_row, text="Validieren", command=self.start_validation, width=80)
        self.btn_validate.pack(side="left", padx=5, pady=(10, 6))

        ctk.CTkButton(button_row, text="AutoAdjust", width=80).pack(side="left", padx=5, pady=(10, 6))

        # Status Label (dieses Label bekommt die Validierungsfarben)
        self.status_label = ctk.CTkLabel(function_frame, text=self._status_text, text_color="#00B7FF")
        self.status_label.pack(pady=(0, 8))

        self.progress_frame = ctk.CTkFrame(function_frame, fg_color="transparent")
        self.progress_frame.pack(pady=6)
        self.progress_frame.pack_forget()

        self.progress = ctk.CTkProgressBar(self.progress_frame, width=300, height=16)
        self.progress.pack(pady=6)
        self.progress.set(0)

        self.progress_text = ctk.CTkLabel(
            self.progress_frame,
            text="0 %",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.progress_text.pack()

        # Am Ende: Status einmal sauber anwenden (damit Theme + Farbe stimmt)
        self._apply_status_style_on_theme_change()

    # =========================
    # 13 ) Sauber schließen
    # =========================
    def _on_close(self):
        self._stop_camera()
        self.destroy()


# ============================
# 14 ) Programmstart
# ============================
if __name__ == "__main__":
    app = App()
    app.mainloop()