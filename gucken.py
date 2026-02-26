# ============================
# 1 ) Bibliotheken importieren
# ============================
import threading
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import shutil
import numpy as np
import time
from datetime import datetime
from PIL import Image, ImageTk

# ============================
#  2 ) Globale Einstellungen
# ============================
appearance = "dark"
colormode = ("Hellmodus")
ctk.set_appearance_mode(appearance)
ctk.set_default_color_theme("blue")

deltatol = 0.01  # Validation: Toleranz (dezimal)

# Globale Farben (werden in _set_live_background überschrieben, aber initialisieren schadet nicht)
val_refcolor = "#00B7FF"
val_loadcolor = "#00B7FF"
val_tolcolor = "#00B7FF"
val_resultcolor = "#00B7FF"


# ============================
# 3 ) Hauptanwendung
# ============================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ===== Fenster Einstellungen =====
        self.title("visCell")
        self.geometry("1440x980")
        self.minsize(1100, 750)

        # ===== Validierung =====
        self.soll_ery = 100
        self.ki_ery = 102
        self.validierung_ok = False

        # ===== Kamera / Video Status-Variablen =====
        self.cap = None
        self.is_streaming = False
        self.current_cam_index = None
        self.last_frame = None                 # immer das "zu exportierende" Bild (BGR)
        self.current_image_path = None

        # ===== Bildanpassung (Helligkeit / Kontrast / Sättigung) =====
        self.base_frame = None                 # "Original" (BGR) für aktuelle Anzeigequelle
        self.contrast_alpha = 1.0              # Kontrastfaktor
        self.brightness_beta = 0.0             # Helligkeitsoffset
        self.saturation_factor = 1.0           # Sättigungsfaktor (1.0 = neutral)

        # Slider-Referenzen
        self.saturation_slider = None
        self.saturation_value_label = None

        self.contrast_slider = None
        self.brightness_slider = None
        self.contrast_value_label = None
        self.brightness_value_label = None

        # ===== Tk-Image Referenz + after()-ID =====
        self._tk_img = None
        self._after_id = None

        # Dropdown Variable (Kamera Auswahl)
        self.camera_var = ctk.StringVar(value="Kamera auswählen...")

        # Freeze Checkbox Variable
        self.freeze_after_capture_var = ctk.BooleanVar(value=False)

        # ===== Speicherordner für Snapshots/Captures =====
        self.capture_dir = "captures"
        os.makedirs(self.capture_dir, exist_ok=True)

        # ===== GUI Widget Referenzen =====
        self.live_view = None
        self.video_container = None
        self.video_label = None
        self.video_text = None

        self.analysis_frame = None
        self.analysis_label = None
        self.cam_status_label = None
        self.darkmode_btn = None

        self.capture_btn = None

        # ===== self.farbwechsel =====
        self.appearance = appearance
        self.colormode_text = colormode

        # ===== Farben für Anzeigezustände =====
        self.placeholder_bg = "#BDBDBD"
        self.live_bg = "black"

        # ===== Metadaten persistent in der App halten =====
        self.metadata = {
            "Analysedatum": "",
            "Prüfer": "",
            "Labor": "",
            "Equipment": "",
            "Vergrößerung": "",
            "Probennummer": "",
            "Notizen": "",
        }

        # Fenster-Schließen abfangen
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ===== GUI bauen =====
        self._build_gui()
        self._build_footer()

        # Dropdown trace
        self.camera_var.trace_add("write", self._camera_var_changed)

        # Initial: Placeholder anzeigen
        self._clear_video_label()

        # Initial Button-Status setzen
        self._update_capture_button_state()

    # =========================
    # Helper: Analyse-Text aus self.metadata bauen
    # =========================
    def _render_analysis_text(self) -> str:
        m = self.metadata
        return (
            "ANALYSE_ERGEBNIS\n\n"
            f"Datum: {m.get('Analysedatum','')}\n"
            f"Prüfer: {m.get('Prüfer','')}\n"
            f"Labor: {m.get('Labor','')}\n\n"
            f"Mikroskop: {m.get('Equipment','')}\n"
            f"Vergrößerung: {m.get('Vergrößerung','')}\n"
            f"Probe Nummer: {m.get('Probennummer','')}\n\n"
            "Erythrozyten Anzahl:      \n"
            "Leukozyten Anzahl:     \n"
            "Hefezellen Anzahl:    \n\n"
            f"Notizen: {m.get('Notizen','')}"
        )

    # =========================
    # 3.1 ) Helper: "Bild aufnehmen" Button aktiv/deaktiv setzen
    # =========================
    def _update_capture_button_state(self):
        choice = self.camera_var.get()
        enable = False

        if isinstance(choice, str) and choice.startswith("Camera "):
            enable = True
        if choice in ("Kamera auswählen...", "Keine Kamera gefunden"):
            enable = False

        if self.capture_btn is not None:
            self.capture_btn.configure(state="normal" if enable else "disabled")

    # ===== Helper: Background für Live/Placeholder setzen =====
    def _set_live_background(self, color: str):
        if self.live_view is not None:
            self.live_view.configure(fg_color=color)
        if self.video_container is not None:
            self.video_container.configure(bg=color)
        if self.video_label is not None:
            self.video_label.configure(bg=color)

        # =========================
        # 4 ) Farbschema Toggle (Hell/Dunkel)
        # =========================
        global val_refcolor, val_loadcolor, val_tolcolor, val_resultcolor

        # Standardfarben: Dunkelmodus
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
    # 5 ) Footer
    # =========================
    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=30)
        footer.pack(fill="x", side="bottom")

        self.darkmode_btn = ctk.CTkButton(footer, text=self.colormode_text, command=self._switchcolor, width=110)
        self.darkmode_btn.pack(side="right", padx=5, pady=5)

    # ===== Kameras finden =====
    def _detect_cameras(self, max_index: int = 5):
        found = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(f"Camera {i}")
            cap.release()
        return found

    # =========================
    # 6 ) Kamera Dropdown via trace
    # =========================
    def _camera_var_changed(self, *_):
        choice = self.camera_var.get()
        self._on_camera_selected(choice)
        self._update_capture_button_state()

    def _on_camera_selected(self, choice: str):
        if choice.startswith("Camera "):
            idx = int(choice.split()[-1])
            self._start_camera(idx)
        else:
            self._stop_camera()
            self._clear_video_label()

    # ===== Freeze Toggle =====
    def _on_freeze_toggle(self):
        if not self.freeze_after_capture_var.get():
            if (self.cap is not None and self.current_cam_index is not None and not self.is_streaming):
                self.is_streaming = True
                self._set_status("Live-Feed fortgesetzt (Freeze aus)")
                self._hide_overlay_text()
                self._set_live_background(self.live_bg)
                self._update_frame_loop()

    # ===== Overlay Handling =====
    def _hide_overlay_text(self):
        try:
            self.video_text.place_forget()
        except Exception:
            pass

    def _show_overlay_text(self, text: str):
        self.video_text.configure(text=text, text_color=("black", "black"))
        self.video_text.place(relx=0.5, rely=0.5, anchor="center")
        self.video_text.lift()

    # ===== Placeholder =====
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

    # ===== Resize helper: FILL =====
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
    # Bildanpassung: Sättigung (HSV S-Kanal)
    # =========================
    def _apply_saturation(self, bgr: np.ndarray, sat: float) -> np.ndarray:
        if bgr is None:
            return None

        # sat: 0.0 .. 2.0 (1.0 = neutral)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        sf = s.astype(np.float32) * float(sat)
        sf = np.clip(sf, 0, 255).astype(np.uint8)

        hsv2 = cv2.merge([h, sf, v])
        out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
        return out

    # =========================
    # Bildanpassung: Kontrast/Helligkeit (LAB L-Kanal)
    # + danach Sättigung (HSV)
    # =========================
    def _apply_brightness_contrast(self, bgr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        if bgr is None:
            return None

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        Lf = L.astype(np.float32)
        Lf = alpha * Lf + beta
        Lf = np.clip(Lf, 0, 255).astype(np.uint8)

        lab2 = cv2.merge([Lf, A, B])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out

    def _apply_all_adjustments(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None:
            return None

        adjusted = self._apply_brightness_contrast(
            bgr,
            alpha=float(self.contrast_alpha),
            beta=float(self.brightness_beta),
        )
        if adjusted is None:
            return None

        adjusted = self._apply_saturation(adjusted, sat=float(self.saturation_factor))
        return adjusted

    def _recompute_and_show_current(self):
        if self.base_frame is None:
            return

        adjusted = self._apply_all_adjustments(self.base_frame)
        if adjusted is None:
            return

        self.last_frame = adjusted
        rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        self._display_rgb_in_video_label(rgb)

    def _on_adjust_changed(self, *_):
        # Sättigung: 0..100 -> sat 0.0..2.0 (50 => 1.0)
        if self.saturation_slider is not None:
            v = float(self.saturation_slider.get())
            self.saturation_factor = max(0.0, v / 50.0)
            if self.saturation_value_label is not None:
                self.saturation_value_label.configure(text=f"{int(v)}%")

        # Kontrast: 0..100 -> alpha ~ 0.01..2.0 (50 => 1.0)
        if self.contrast_slider is not None:
            v = float(self.contrast_slider.get())
            self.contrast_alpha = max(0.01, v / 50.0)
            if self.contrast_value_label is not None:
                self.contrast_value_label.configure(text=f"{int(v)}%")

        # Helligkeit: 0..100 -> beta -100..+100 (50 => 0)
        if self.brightness_slider is not None:
            v = float(self.brightness_slider.get())
            self.brightness_beta = (v - 50.0) * 2.0
            if self.brightness_value_label is not None:
                self.brightness_value_label.configure(text=f"{int(v)}%")

        self._recompute_and_show_current()

    def _auto_adjust(self, target_mean: float = 150.0, target_std: float = 55.0):
        if self.base_frame is None:
            self._set_status("AutoAdjust: Kein Bild vorhanden")
            return

        lab = cv2.cvtColor(self.base_frame, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)

        mean = float(np.mean(L))
        std = float(np.std(L))
        if std < 1e-6:
            std = 1e-6

        alpha = target_std / std
        beta = target_mean - alpha * mean

        alpha = float(np.clip(alpha, 0.01, 3.0))
        beta = float(np.clip(beta, -120.0, 120.0))

        self.contrast_alpha = alpha
        self.brightness_beta = beta

        # inverse Mapping zu Slidern
        v_contrast = int(np.clip(round(alpha * 50.0), 0, 100))
        v_brightness = int(np.clip(round(beta / 2.0 + 50.0), 0, 100))

        if self.contrast_slider is not None:
            self.contrast_slider.set(v_contrast)
        if self.brightness_slider is not None:
            self.brightness_slider.set(v_brightness)

        # Sättigung lassen wir bei AutoAdjust unverändert (nur Kontrast/Helligkeit)
        self._on_adjust_changed()
        self._set_status(f"AutoAdjust: alpha={alpha:.2f}, beta={beta:.1f}")

    # =========================
    # 7 ) Live Loop
    # =========================
    def _update_frame_loop(self):
        if not self.is_streaming or self.cap is None:
            return

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)

                self.current_image_path = None

                # Base = Roh-Frame
                self.base_frame = frame.copy()

                # Angepasst berechnen und anzeigen (inkl. Sättigung)
                adjusted = self._apply_all_adjustments(self.base_frame)
                self.last_frame = adjusted

                rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)

                # Widgetgröße holen
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

    # ===== Snapshot =====
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

                # Base neu setzen (aus Datei)
                self.base_frame = bgr.copy()

                # Anpassung anwenden + anzeigen (setzt auch last_frame)
                self._recompute_and_show_current()

                self._set_status(f"Gespeichert & geöffnet (Freeze): {path}")

            except Exception as e:
                rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                self._display_rgb_in_video_label(rgb)
                self._set_status(f"Gespeichert (Freeze) - Laden fehlgeschlagen: {e}")
        else:
            self._set_status(f"Gespeichert: {path}")

    # =========================
    # 8 ) Bild importieren
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

            # Base setzen (Original)
            self.base_frame = bgr.copy()
            self.current_image_path = file_path

            # Angepasst anzeigen (setzt last_frame)
            self._recompute_and_show_current()

            self._set_status(f"Importiert: {os.path.basename(file_path)}")

        except Exception as e:
            self._set_status("Import fehlgeschlagen")
            messagebox.showerror("Fehler", f"Bild konnte nicht importiert werden:\n{e}")

    # =========================
    # 9 ) Bild exportieren
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

    # ===== Optional: Alles aus captures/ exportieren =====
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

    # ===== Status Text in GUI setzen =====
    def _set_status(self, msg: str):
        if self.cam_status_label is not None:
            self.cam_status_label.configure(text=msg)

    # =========================
    # 10 ) Metadaten Popup (Formular) - mit Persistenz & Vorbefüllung
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

        # Vorbelegung mit gespeicherten Werten
        for field in fields:
            frame = ctk.CTkFrame(popup)
            frame.pack(fill="x", padx=20, pady=5)

            label = ctk.CTkLabel(frame, text=field, width=100, anchor="w")
            label.pack(side="left")

            entry = ctk.CTkEntry(frame)
            entry.pack(side="right", fill="x", expand=True)

            entry.insert(0, str(self.metadata.get(field, "")))
            entries[field] = entry

        def save_metadata():
            analysedatum_raw = entries["Analysedatum"].get().strip()
            if analysedatum_raw == "":
                analysedatum_raw = datetime.now().strftime("%d.%m.%Y %H:%M")

            # in App-State schreiben (bleibt erhalten)
            self.metadata["Analysedatum"] = analysedatum_raw
            self.metadata["Prüfer"] = entries["Prüfer"].get().strip()
            self.metadata["Labor"] = entries["Labor"].get().strip()
            self.metadata["Equipment"] = entries["Equipment"].get().strip()
            self.metadata["Vergrößerung"] = entries["Vergrößerung"].get().strip()
            self.metadata["Probennummer"] = entries["Probennummer"].get().strip()
            self.metadata["Notizen"] = entries["Notizen"].get().strip()

            analysis_label.configure(text=self._render_analysis_text())
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
    # 12 ) GUI bauen
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

        self.capture_btn = ctk.CTkButton(bottom_controls, text="Bild aufnehmen", command=self._save_snapshot)
        self.capture_btn.grid(row=0, column=2, padx=5)

        ctk.CTkButton(bottom_controls, text="Alle Captures exportieren", command=self._export_all_captures).grid(row=0, column=3, padx=5)

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
            text=self._render_analysis_text(),
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

        # ===== Sättigung (JETZT: wirkt wirklich aufs Bild) =====
        row = ctk.CTkFrame(slider_frame, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(row, text="Sättigung").pack(side="left")
        value_label = ctk.CTkLabel(row, text="50%")
        value_label.pack(side="right")

        saturation_slider = ctk.CTkSlider(slider_frame, from_=0, to=100)
        saturation_slider.pack(fill="x", padx=10, pady=(0, 10))

        self.saturation_slider = saturation_slider
        self.saturation_value_label = value_label

        saturation_slider.configure(command=self._on_adjust_changed)
        saturation_slider.set(50)

        # ===== Kontrast =====
        row_contrast = ctk.CTkFrame(slider_frame, fg_color="transparent")
        row_contrast.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(row_contrast, text="Kontrast").pack(side="left")
        contrast_value_label = ctk.CTkLabel(row_contrast, text="50%")
        contrast_value_label.pack(side="right")

        contrast_slider = ctk.CTkSlider(slider_frame, from_=0, to=100)
        contrast_slider.pack(fill="x", padx=10, pady=(0, 10))

        self.contrast_slider = contrast_slider
        self.contrast_value_label = contrast_value_label

        contrast_slider.configure(command=self._on_adjust_changed)
        contrast_slider.set(50)

        # ===== Helligkeit =====
        row_brightness = ctk.CTkFrame(slider_frame, fg_color="transparent")
        row_brightness.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(row_brightness, text="Helligkeit").pack(side="left")
        brightness_value_label = ctk.CTkLabel(row_brightness, text="50%")
        brightness_value_label.pack(side="right")

        brightness_slider = ctk.CTkSlider(slider_frame, from_=0, to=100)
        brightness_slider.pack(fill="x", padx=10, pady=(0, 10))

        self.brightness_slider = brightness_slider
        self.brightness_value_label = brightness_value_label

        brightness_slider.configure(command=self._on_adjust_changed)
        brightness_slider.set(50)

        # Initiale Labels/Parameter korrekt setzen (einmal triggern)
        self._on_adjust_changed()

        # ===== Buttons unten =====
        function_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        function_frame.pack(fill="x", padx=10, pady=10)

        button_row = ctk.CTkFrame(function_frame, fg_color="transparent")
        button_row.pack(fill="x")

        self.btn_validate = ctk.CTkButton(button_row, text="Validieren", command=self.start_validation, width=80)
        self.btn_validate.pack(side="left", padx=5, pady=(10, 6))

        ctk.CTkButton(button_row, text="AutoAdjust", command=self._auto_adjust, width=80)\
            .pack(side="left", padx=5, pady=(10, 6))

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