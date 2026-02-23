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
import shutil  # Dateikopien (copy2) für "Export all"
import numpy as np  # Bilddaten als Arrays (RGB/BGR) bearbeiten
import time
from datetime import datetime  # Zeitstempel für Dateinamen
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
        self.geometry("1440x980")  # Startgröße
        self.minsize(1100, 750)  # Mindestgröße (User kann nicht kleiner ziehen)

        # ===== Validierung =====
        # DUMMY-Analyse
        self.soll_ery = 100  # Referenzwert (Soll)
        self.ki_ery = 102  # Dummy-KI-Ausgabe
        self.validierung_ok = False  # Validierung ok/nok

        # ===== Kamera / Video Status-Variablen =====
        self.cap = None  # OpenCV VideoCapture Objekt (None = keine Kamera geöffnet)
        self.is_streaming = (False)  # Flag: True = Live-Loop soll laufen, False = kein Loop
        self.current_cam_index = None  # gemerkter Index der aktiven Kamera (0/1/2/...)
        self.last_frame = None  # letzter Frame als BGR (OpenCV Standardformat)
        self.current_image_path = (None)  # Pfad des aktuell importierten/angezeigten Bildes (falls vorhanden)

        # ===== Tk-Image Referenz + after()-ID =====
        self._tk_img = (None)  # Referenz auf PhotoImage (wichtig: sonst zeigt Tk das Bild nicht)
        self._after_id = (None)  # ID des geplanten self.after(...) Calls (zum sauberen Abbrechen)

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

        # NEU: Button-Referenz "Bild aufnehmen"
        self.capture_btn = None

        # ===== self.farbwechsel =====
        self.appearance = appearance
        self.colormode_text = colormode

        # ===== Farben für Anzeigezustände =====
        self.placeholder_bg = "#BDBDBD"  # hellgrau: wenn kein Live-Feed/Bild
        self.live_bg = "black"  # schwarz: wenn Live-Feed läuft / Bild angezeigt wird

        # Fenster-Schließen abfangen (damit Kamera sauber freigegeben wird)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ===== GUI bauen =====
        self._build_gui()
        self._build_footer()

        # Dropdown trace: wenn camera_var geändert wird -> Kamerawechsel ausführen
        self.camera_var.trace_add("write", self._camera_var_changed)

        # Initial: Placeholder anzeigen
        self._clear_video_label()

        # NEU: Initial den Zustand vom "Bild aufnehmen" Button setzen
        self._update_capture_button_state()

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

        # Standard: deaktiviert, bis eine echte Kamera gewählt ist
        enable = False

        if isinstance(choice, str) and choice.startswith("Camera "):
            enable = True

        if choice in ("Kamera auswählen...", "Keine Kamera gefunden"):
            enable = False

        if self.capture_btn is not None:
            self.capture_btn.configure(state="normal" if enable else "disabled")

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

        # =========================
        # 4 ) Farbschema Toggle (Hell/Dunkel)
        # =========================
        # ===== Toggle: Validation =====
        global val_refcolor, val_loadcolor, val_tolcolor, val_resultcolor

        # ===== Standardfarben: Dunkelmodus =====
        val_refcolor = "#00B7FF"
        val_loadcolor = "#00B7FF"
        val_tolcolor = "#00B7FF"
        val_resultcolor = "#00B7FF"

        match appearance:
            case "dark":
                self.status_label.configure(text_color="#00B7FF")
                val_refcolor = "#00B7FF"
            case "light":
                self.status_label.configure(text_color="#000000")
                val_refcolor = "#000000"
                val_loadcolor = "#000000"
                val_tolcolor = "#000000"
                val_resultcolor = "#000000"
            case _:
                self.status_label.configure(text_color="#FF0000")

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

    # =========================
    # 5 ) Footer (unten rechts: Theme Toggle Button)
    # =========================
    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=30)  # Footer-Leiste
        footer.pack(fill="x", side="bottom")

        # ===== Button: Farbwechsel =====
        self.darkmode_btn = ctk.CTkButton(footer, text=self.colormode_text, command=self._switchcolor, width=110)
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
    # Wird automatisch aufgerufen wenn camera_var sich ändert
    def _camera_var_changed(self, *_):
        choice = self.camera_var.get()  # aktuellen Auswahltext holen
        self._on_camera_selected(choice)  # Auswahl behandeln

        # NEU: Nach jeder Dropdown-Änderung den Button-Status aktualisieren
        self._update_capture_button_state()

    # Falls "Camera X" gewählt -> Kamera starten
    def _on_camera_selected(self, choice: str):
        if choice.startswith("Camera "):
            idx = int(choice.split()[-1])  # Index aus "Camera 0" etc. extrahieren
            self._start_camera(idx)  # Kamera starten
        else:
            # Falls keine echte Kamera gewählt -> stoppen + Placeholder
            self._stop_camera()
            self._clear_video_label()

    # ===== Freeze Toggle =====
    def _on_freeze_toggle(self):
        # Diese Funktion wird aufgerufen wenn der Freeze-Checkbox-Status verändert wird
        # Hier: Wenn Freeze AUSgeschaltet wird und eine Kamera offen ist, soll Live wieder laufen
        if not self.freeze_after_capture_var.get():  # Freeze = False
            # Nur weiterlaufen, wenn Kamera offen ist und wir gerade NICHT streamen
            if (self.cap is not None and self.current_cam_index is not None and not self.is_streaming):
                self.is_streaming = True
                self._set_status("Live-Feed fortgesetzt (Freeze aus)")
                self._hide_overlay_text()
                self._set_live_background(self.live_bg)
                self._update_frame_loop()  # Live-Loop erneut starten

    # ===== Overlay Handling (Text über dem Video) =====
    def _hide_overlay_text(self):
        try:  # Overlay-Label ausblenden (wenn es existiert)
            self.video_text.place_forget()
        except Exception:
            pass

    # ===== Overlay-Text setzen + zentriert anzeigen =====
    def _show_overlay_text(self, text: str):
        self.video_text.configure(text=text, text_color=("black", "black"))  # tuple: für hell/dunkel robust
        self.video_text.place(relx=0.5, rely=0.5, anchor="center")
        self.video_text.lift()  # in den Vordergrund

    # =====Placeholder Hintergrund aktivieren =====
    def _clear_video_label(self):
        self._set_live_background(self.placeholder_bg)

        # Bildreferenz löschen (damit nichts angezeigt wird)
        self._tk_img = None
        if self.video_label is not None:
            self.video_label.configure(image="")
            self.video_label.image = None

        # Overlay Text anzeigen: "Kein Bild..."
        if self.video_text is not None:
            self._show_overlay_text("Kein Bild vorhanden\n(bitte Kamera auswählen)")

        # Tkinter updaten (stellt sicher dass UI Änderungen "sofort" sichtbar sind)
        self.update_idletasks()

    # ===== Kamera Start/Stop =====
    def _start_camera(self, index: int):
        self._stop_camera()  # Erst alles stoppen (inkl. after cancel), damit nichts parallel läuft
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        # ===== Wenn Kamera nicht geöffnet werden kann -> Abbruch und Hinweis =====
        if not self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            self._set_live_background(self.placeholder_bg)
            self._show_overlay_text(f"Kamera {index} konnte nicht geöffnet werden")

            # NEU: Button-Status aktualisieren (bleibt deaktiviert, wenn keine Kamera wirklich läuft)
            self._update_capture_button_state()
            return

        # ===== Kamera erfolgreich: Status setzen =====
        self.current_cam_index = index
        self.is_streaming = True

        self._set_live_background(self.live_bg)  # Live Hintergrund setzen
        self._hide_overlay_text()  # Overlay ausblenden
        self._update_frame_loop()  # Live-Loop starten (Frames regelmäßig lesen und anzeigen)

        # NEU: Button-Status aktualisieren (jetzt aktiv)
        self._update_capture_button_state()

    # ===== Streaming deaktivieren =====
    def _stop_camera(self):
        self.is_streaming = False
        self.current_cam_index = None

        # after-loop abbrechen, falls aktiv
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        # Kamera freigeben
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Tk-Image Referenz löschen
        self._tk_img = None

        # Background wieder placeholder
        self._set_live_background(self.placeholder_bg)

        # NEU: Button-Status aktualisieren (deaktiviert)
        self._update_capture_button_state()

    # ===== Resize helper: FILL (Panel füllen, ggf. Crop) =====
    def _resize_fill(self, rgb, target_w, target_h):
        # Originalmaße aus Array holen
        h, w = rgb.shape[:2]

        # Schutz: ungültige Größe
        if w <= 0 or h <= 0:
            return rgb

        # Scale wählen: so groß, dass Ziel komplett gefüllt ist (max statt min)
        scale = max(target_w / w, target_h / h)

        # Neue Größe berechnen
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # Skalieren
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center Crop: mittig das Ziel ausschneiden
        x0 = (new_w - target_w) // 2
        y0 = (new_h - target_h) // 2

        return resized[y0: y0 + target_h, x0: x0 + target_w]

    # ===== Helper: RGB Bild im Panel anzeigen =====
    def _display_rgb_in_video_label(self, rgb: np.ndarray):
        """
        Zeigt ein RGB numpy array im linken Video-Label an (mit Fill-Resize).
        Wichtig: rgb muss RGB sein (nicht BGR), weil PIL/Anzeige so erwartet wird.
        """
        # Falls GUI noch nicht gebaut/Label nicht vorhanden -> raus
        if self.video_label is None:
            return

        # ===== Hintergrund + Overlay =====
        self._set_live_background(self.live_bg)
        self._hide_overlay_text()

        # ===== Aktuelle Widgetgröße holen (damit wir passend skalieren) =====
        self.video_label.update_idletasks()
        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()

        # Fallback falls Widget noch nicht richtig gerendert ist
        if target_w <= 50 or target_h <= 50:
            target_w, target_h = 1280, 720

        # Bild auf Ziel "fillen" (crop möglich)
        rgb = self._resize_fill(rgb, target_w, target_h)

        # NumPy -> PIL
        pil_img = Image.fromarray(rgb)

        # Exakt auf Zielgröße (zur Sicherheit)
        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # PIL -> Tk PhotoImage (muss als Referenz gespeichert werden)
        self._tk_img = ImageTk.PhotoImage(pil_img)

        # In Label anzeigen
        self.video_label.configure(image=self._tk_img)
        self.video_label.image = self._tk_img  # zweite Referenz, Tk ist manchmal zickig

    # =========================
    # 7 ) Live Loop (liest Frames und zeigt sie)
    # =========================
    def _update_frame_loop(self):
        # Wenn kein streaming aktiv oder cap fehlt -> nichts tun
        if not self.is_streaming or self.cap is None:
            return

        try:
            ret, frame = self.cap.read()  # Frame aus Kamera lesen
            if ret and frame is not None:  # Nur wenn Frame gültig ist
                frame = cv2.flip(frame, 1)  # Spiegelung (wirkt natürlicher wie Selfie)
                self.last_frame = frame  # Frame als "letztes Bild" speichern (OpenCV = BGR)
                self.current_image_path = None  # Wenn wir live sind, ist das kein importiertes Bild mehr
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB (für Anzeige)

                # Widgetgröße holen
                self.video_label.update_idletasks()
                target_w = self.video_label.winfo_width()
                target_h = self.video_label.winfo_height()

                if target_w > 50 and target_h > 50:
                    rgb = self._resize_fill(rgb, target_w, target_h)  # Bild passend füllen

                    # PIL für genaue Skalierung
                    pil_img = Image.fromarray(rgb)
                    pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
                else:
                    pil_img = Image.fromarray(rgb)  # Fallback: ohne exakte Größe

                # Tk Bild erzeugen und anzeigen
                self._tk_img = ImageTk.PhotoImage(pil_img)
                self.video_label.configure(image=self._tk_img)
                self.video_label.image = self._tk_img

        except Exception as e:
            print("Fehler im Live-Loop:", e)  # Debug: Fehler im Loop ausgeben

        self._after_id = self.after(33, self._update_frame_loop)  # Loop erneut planen (ca. 30 FPS)

    # ===== Snapshot (speichert in captures/) =====
    def _save_snapshot(self):
        # Ohne last_frame gibt es nichts zu speichern
        if self.last_frame is None:
            self._set_status("Kein Frame vorhanden (Kamera läuft? / Bild importiert?)")
            return

        # Zeitstempel erzeugen (ms genau)
        ts = datetime.now().strftime("%d%m%Y_%H%M%S_%f")[:-3]

        # Dateiname enthält Kameraindex oder "img", falls kein Kameraindex vorhanden
        cam = (f"cam{self.current_cam_index}" if self.current_cam_index is not None else "img")
        filename = f"snapshot_{cam}_{ts}.png"

        path = os.path.join(self.capture_dir, filename)  # Speicherpfad (captures/filename)
        ok = cv2.imwrite(path, self.last_frame)  # Speichern mit OpenCV (BGR)

        # Wenn Speichern fehlschlägt -> Abbruch
        if not ok:
            self._set_status("Speichern fehlgeschlagen")
            return

        # ===== Freeze aktiv: Live stoppen + Bild aus Datei öffnen und anzeigen =====
        if self.freeze_after_capture_var.get() and self.cap is not None:
            self.is_streaming = False  # Live-Loop stoppen (damit keine neuen Frames mehr angezeigt werden)

            # geplanten after-loop abbrechen, damit garantiert nichts mehr läuft
            if self._after_id is not None:
                try:
                    self.after_cancel(self._after_id)
                except Exception:
                    pass
                self._after_id = None

            self.current_image_path = path  # Pfad des aktuell angezeigten Bildes merken

            # ===== Das gespeicherte Bild wirklich aus der Datei laden (so wie "öffnen") =====
            try:
                pil_img = Image.open(path).convert("RGB")  # Datei -> PIL -> RGB
                rgb = np.array(pil_img)  # PIL -> numpy RGB
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # zurück nach BGR (für spätere Exporte)
                self.last_frame = bgr  # last_frame aktualisieren

                self._display_rgb_in_video_label(rgb)  # Bild im linken Panel anzeigen
                self._set_status(f"Gespeichert & geöffnet (Freeze): {path}")

            # ===== Fallback: wenn Datei laden schiefgeht -> wenigstens den Frame anzeigen =====
            except Exception as e:
                rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                self._display_rgb_in_video_label(rgb)
                self._set_status(f"Gespeichert (Freeze) - Laden fehlgeschlagen: {e}")
        else:
            self._set_status(f"Gespeichert: {path}")  # Freeze aus: nur speichern und Status setzen

    # =========================
    # 8 ) Bild importieren (Dialog startet in captures/)
    # =========================
    def _import_image(self):
        # Kamera anhalten, damit Live nicht direkt das importierte Bild überschreibt
        self._stop_camera()

        # ===== Datei-Dialog zum Öffnen eines Bildes =====
        file_path = filedialog.askopenfilename(
            title="Bild importieren",
            initialdir=self.capture_dir,  # startet im captures Ordner
            filetypes=[
                ("Bilddateien", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Alle Dateien", "*.*"),
            ],
        )

        # Wenn Benutzer abbricht -> Status
        if not file_path:
            self._set_status("Import abgebrochen")
            return

        try:
            # Bild laden und in RGB umwandeln
            pil_img = Image.open(file_path).convert("RGB")
            rgb = np.array(pil_img)  # RGB numpy
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # BGR für OpenCV-Export

            self.last_frame = bgr  # last_frame setzen (damit Export/Save möglich ist)
            self.current_image_path = file_path  # Pfad merken
            self._display_rgb_in_video_label(rgb)  # Bild anzeigen
            self._set_status(f"Importiert: {os.path.basename(file_path)}")  # Status

        # Fehler -> Status + Popup
        except Exception as e:
            self._set_status("Import fehlgeschlagen")
            messagebox.showerror("Fehler", f"Bild konnte nicht importiert werden:\n{e}")

    # =========================
    # 9 ) Bild exportieren (Save-Dialog; startet in captures/)
    # =========================
    def _export_image(self):
        # Ohne last_frame gibt es nichts zu exportieren
        if self.last_frame is None:
            self._set_status("Kein Bild zum Exportieren vorhanden")
            return

        # Default Dateiname
        ts = datetime.now().strftime("%d%m%Y_%H%M%S")
        default_name = f"export_{ts}.png"

        # Speichern-Dialog
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

        # Abbruch
        if not save_path:
            self._set_status("Export abgebrochen")
            return

        try:
            ok = cv2.imwrite(save_path, self.last_frame)  # OpenCV speichern (last_frame = BGR)

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

        # Abbruch
        if not target_dir:
            self._set_status("Export abgebrochen")
            return

        try:
            count = 0

            # Alle Dateien im captures Ordner durchgehen
            for name in os.listdir(self.capture_dir):
                src = os.path.join(self.capture_dir, name)

                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(target_dir, name))
                    count += 1

            self._set_status(f"Exportiert: {count} Datei(en) nach {target_dir}")

        except Exception as e:
            self._set_status("Export aller Captures fehlgeschlagen")
            messagebox.showerror("Fehler", f"Konnte Captures nicht exportieren:\n{e}")

    # ===== Status Text in GUI setzen =====
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
    # 11 ) Validierung
    # =========================
    def start_validation(self):
        self.btn_validate.configure(state="disabled")
        self._show_progress()
        thread = threading.Thread(target=self._validation_process, daemon=True)
        thread.start()

    def _validation_process(self):
        time.sleep(0.4)

        self._status("Lade Referenzdaten...", val_refcolor)
        self._update_progress(1, 5)
        time.sleep(0.1)

        soll = int(self.soll_ery)

        self._status("KI analysiert Referenzbild...", val_loadcolor)
        self._update_progress(2, 5)
        time.sleep(0.8)

        ist = int(self.ki_ery)

        self._status("Prüfe Ergebnis (Regel A)...", val_tolcolor)
        self._update_progress(3, 5)
        time.sleep(0.05)

        ok, tol = self._regel_check(ist=ist, soll=soll)
        self.validierung_ok = bool(ok)

        self._status("Ergebnis bereit (Bediener bestätigen)...", val_resultcolor)
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

        if appearance == "dark":
            val_yes = "#00FF66"
            val_unknown = "#FFAA00"
            val_no = "#FF4444"
        else:
            val_yes = "#005823"
            val_unknown = "#A86D00"
            val_no = "#DB0101"

        if confirmed and self.validierung_ok:
            self._status("Validierung bestätigt ✅", val_yes)
        elif confirmed and not self.validierung_ok:
            self._status("Bestätigt, aber NICHT bestanden ⚠️", val_unknown)
        else:
            self._status("Nicht bestätigt ❌", val_no)

        self.btn_validate.configure(state="normal")

    def _status(self, text: str, farbe: str):
        self.after(0, lambda: self.status_label.configure(text=text, text_color=farbe))

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

        self.camera_select = ctk.CTkOptionMenu(controls_frame, values=camera_values, variable=self.camera_var)
        self.camera_select.grid(row=0, column=0, padx=5)

        self.freeze_checkbox = ctk.CTkCheckBox(
            controls_frame,
            text="Nach Aufnahme pausieren",
            variable=self.freeze_after_capture_var,
            command=self._on_freeze_toggle
        )
        self.freeze_checkbox.grid(row=0, column=1, padx=10)

        ctk.CTkButton(controls_frame, text="Analyse starten").grid(row=0, column=2, padx=5)

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

        # =========================
        # 12.1 ) Button: "Bild aufnehmen" (mit disabled/normal)
        # =========================
        # NEU:
        # - Wir speichern eine Referenz (self.capture_btn),
        #   damit wir den Button später ausgrauen/aktivieren können.
        self.capture_btn = ctk.CTkButton(bottom_controls, text="Bild aufnehmen", command=self._save_snapshot)
        self.capture_btn.grid(row=0, column=2, padx=5)

        ctk.CTkButton(bottom_controls, text="Alle Captures exportieren", command=self._export_all_captures).grid(row=0, column=3, padx=5)

        # NEU: Direkt nach dem Erstellen einmal den Zustand setzen
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

        self.status_label = ctk.CTkLabel(function_frame, text="Bereit", text_color="#00B7FF")
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