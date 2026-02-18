"""
Modul: gui.py
Beschreibung:
Dieses Skript ist die grafische Benutzeroberflächedes Projekts Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse von mikroskopischen Zellstrukturen.

Autor: Max Sielhorst
Co-Autor: Marlon Aust für Hell-/Dunkelmodus, Validierungsfunktion
Co-Autor: Sven Klapp für Error-Handling
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""


# ============================
# 1 ) Bibliotheken importieren
# ============================

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import shutil
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk

# NEU: Windows Device-Events (WM_DEVICECHANGE) statt Polling
import sys
import ctypes
from ctypes import wintypes

# ============================
#  Globale Einstellungen
# ============================

appearance = "dark"
colormode = "Hellmodus"
ctk.set_appearance_mode(appearance)
ctk.set_default_color_theme("blue")

# ============================
#   Hauptanwendung
# ============================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Fenster
        self.title("visCell")
        self.geometry("1440x980")
        self.minsize(1100, 750)

        # ===== Kamera / Video =====
        self.cap = None
        self.is_streaming = False
        self.current_cam_index = None
        self.last_frame = None  # BGR numpy-array (für cv2.imwrite)
        self.current_image_path = None  # merkt sich importiertes/angezeigtes Bild

        # Tk Image Referenz + after-id zum sicheren Stop
        self._tk_img = None
        self._after_id = None

        # ===== Kamera Dropdown State =====
        self._last_camera_values = []

        # Debounce: DeviceChange kann mehrfach feuern -> wir refreshen verzögert einmal
        self._wm_refresh_after_id = None

        # ===== Windows DeviceChange Hook (statt Polling) =====
        self._wm_old_proc = None
        self._wm_new_proc = None
        self._SetWindowLongPtr = None
        self._CallWindowProc = None

        # Dropdown Variable
        self.camera_var = ctk.StringVar(value="Kamera auswählen...")

        # Freeze nach Aufnahme
        self.freeze_after_capture_var = ctk.BooleanVar(value=False)

        # Speicherordner
        self.capture_dir = "captures"
        os.makedirs(self.capture_dir, exist_ok=True)

        # GUI-Referenzen
        self.live_view = None
        self.video_container = None
        self.video_label = None
        self.video_text = None

        self.analysis_frame = None
        self.analysis_label = None
        self.status_label = None
        self.darkmode_btn = None

        self.camera_select = None  # OptionMenu Referenz

        # "Bild aufnehmen" Button Referenz (für disabled/normal)
        self.capture_btn = None

        # Legacy Variablen
        self.appearance = appearance
        self.colormode_text = colormode

        # Farben für Live/Placeholder
        self.placeholder_bg = "#BDBDBD"
        self.live_bg = "black"

        # Close handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # GUI bauen
        self._build_gui()
        self._build_footer()

        # Placeholder initial
        self._clear_video_label()

        # Initial: Button-Status passend zur Default-Auswahl setzen
        self._update_capture_button_state()

        # Initiale Kameraliste setzen
        self._last_camera_values = self._get_camera_menu_values()
        if self.camera_select is not None:
            self.camera_select.configure(values=self._last_camera_values)

        # Statt Polling: auf Windows Geräte-Events hören (USB Kamera an/ab)
        self._install_device_change_hook()

    # =========================
    # Helper: Background für Live/Placeholder setzen
    # =========================
    def _set_live_background(self, color: str):
        # CTkFrame
        if self.live_view is not None:
            self.live_view.configure(fg_color=color)
        # Tk container/label
        if self.video_container is not None:
            self.video_container.configure(bg=color)
        if self.video_label is not None:
            self.video_label.configure(bg=color)

    # =========================
    # "Bild aufnehmen" Button aktiv/deaktiv
    # =========================
    def _update_capture_button_state(self):
        choice = self.camera_var.get()
        should_enable = isinstance(choice, str) and choice.startswith("Camera ")
        if choice in ("Kamera auswählen...", "Keine Kamera gefunden"):
            should_enable = False
        if self.capture_btn is not None:
            self.capture_btn.configure(state="normal" if should_enable else "disabled")

    # =========================
    # Farbschema Toggle
    # =========================
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

        # Background nach Themewechsel erneut korrekt setzen
        if self.is_streaming:
            self._set_live_background(self.live_bg)
        else:
            self._set_live_background(self.placeholder_bg)

    # =========================
    # Footer
    # =========================
    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=30)
        footer.pack(fill="x", side="bottom")

        self.darkmode_btn = ctk.CTkButton(
            footer,
            text=self.colormode_text,
            command=self._switchcolor,
            width=110,
        )
        self.darkmode_btn.pack(side="right", padx=5, pady=5)

    # =========================
    # Windows: WM_DEVICECHANGE Hook
    # =========================
    def _install_device_change_hook(self):
        """
        Windows-only: Hookt WM_DEVICECHANGE, damit wir das Dropdown aktualisieren,
        wenn Geräte (z.B. USB-Kamera) an/abgesteckt werden.
        """
        if not sys.platform.startswith("win"):
            return

        self.update_idletasks()
        hwnd = self.winfo_id()

        GWL_WNDPROC = -4
        WM_DEVICECHANGE = 0x0219

        # 64bit vs 32bit
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            SetWindowLongPtrW = ctypes.windll.user32.SetWindowLongPtrW
            CallWindowProcW = ctypes.windll.user32.CallWindowProcW
            self._SetWindowLongPtr = SetWindowLongPtrW
            self._CallWindowProc = CallWindowProcW
            self._SetWindowLongPtr.restype = ctypes.c_void_p
            self._SetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
            self._CallWindowProc.restype = ctypes.c_longlong
            self._CallWindowProc.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            CALLBACK_RESTYPE = ctypes.c_longlong
        else:
            SetWindowLongW = ctypes.windll.user32.SetWindowLongW
            CallWindowProcW = ctypes.windll.user32.CallWindowProcW
            self._SetWindowLongPtr = SetWindowLongW
            self._CallWindowProc = CallWindowProcW
            self._SetWindowLongPtr.restype = ctypes.c_long
            self._SetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
            self._CallWindowProc.restype = ctypes.c_long
            self._CallWindowProc.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            CALLBACK_RESTYPE = ctypes.c_long

        WNDPROC = ctypes.WINFUNCTYPE(CALLBACK_RESTYPE, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        def _proc(hWnd, msg, wParam, lParam):
            if msg == WM_DEVICECHANGE:
                self._schedule_camera_refresh()
            return self._CallWindowProc(self._wm_old_proc, hWnd, msg, wParam, lParam)

        self._wm_new_proc = WNDPROC(_proc)
        self._wm_old_proc = self._SetWindowLongPtr(hwnd, GWL_WNDPROC, self._wm_new_proc)

    def _uninstall_device_change_hook(self):
        if not sys.platform.startswith("win"):
            return
        if self._wm_old_proc is None:
            return
        hwnd = self.winfo_id()
        GWL_WNDPROC = -4
        try:
            self._SetWindowLongPtr(hwnd, GWL_WNDPROC, self._wm_old_proc)
        except Exception:
            pass
        self._wm_old_proc = None
        self._wm_new_proc = None

    def _schedule_camera_refresh(self):
        # Debounce: viele Events in kurzer Folge -> einmal verzögert refreshen
        if self._wm_refresh_after_id is not None:
            try:
                self.after_cancel(self._wm_refresh_after_id)
            except Exception:
                pass
            self._wm_refresh_after_id = None

        # Kurze Wartezeit, bis Windows "fertig" ist
        self._wm_refresh_after_id = self.after(400, self._refresh_camera_dropdown_once)

    def _refresh_camera_dropdown_once(self):
        self._wm_refresh_after_id = None

        new_values = self._get_camera_menu_values()
        if new_values == self._last_camera_values:
            self._update_capture_button_state()
            return

        self._last_camera_values = new_values

        if self.camera_select is not None:
            self.camera_select.configure(values=new_values)

        # Wenn laufende Kamera entfernt wurde -> stoppen und reset
        active_choice = f"Camera {self.current_cam_index}" if self.current_cam_index is not None else None
        if active_choice is not None and active_choice not in new_values:
            self._stop_camera()
            self._clear_video_label()
            self.camera_var.set("Kamera auswählen...")

        # Wenn keine Kamera läuft, aber ungültige Auswahl -> reset
        current_choice = self.camera_var.get()
        if self.current_cam_index is None and current_choice not in new_values:
            self.camera_var.set("Kamera auswählen...")

        self._update_capture_button_state()

    # =========================
    # Kameras finden (aktive Kamera NICHT anfassen)
    # =========================
    def _detect_cameras(self, max_index: int = 5, skip_index=None):
        found = []
        for i in range(max_index + 1):
            # Aktive Kamera beim Scannen NICHT anfassen
            if skip_index is not None and i == skip_index:
                if self.is_streaming and self.cap is not None:
                    found.append(f"Camera {i}")
                continue

            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(f"Camera {i}")
            cap.release()

        return found

    def _get_camera_menu_values(self):
        skip = self.current_cam_index if (self.is_streaming and self.current_cam_index is not None) else None
        detected = self._detect_cameras(max_index=5, skip_index=skip)

        camera_values = ["Kamera auswählen..."] + detected
        if len(detected) == 0:
            camera_values = ["Kamera auswählen...", "Keine Kamera gefunden"]
        return camera_values

    # =========================
    # Kamera Auswahl (nur bei User-Klick)
    # =========================
    def _on_camera_selected(self, choice: str):
        if choice.startswith("Camera "):
            idx = int(choice.split()[-1])

            # Wenn dieselbe Kamera schon läuft: nichts neu starten
            if self.current_cam_index == idx and self.is_streaming and self.cap is not None:
                self._update_capture_button_state()
                return

            self._start_camera(idx)
        else:
            self._stop_camera()
            self._clear_video_label()

            # Wenn User zurück auf "Kamera auswählen..." geht, einmal aktualisieren
            self._refresh_camera_dropdown_once()

        self._update_capture_button_state()

    # =========================
    # Freeze Toggle
    # =========================
    def _on_freeze_toggle(self):
        if not self.freeze_after_capture_var.get():
            if self.cap is not None and self.current_cam_index is not None and not self.is_streaming:
                self.is_streaming = True
                self._set_status("Live-Feed fortgesetzt (Freeze aus)")
                self._hide_overlay_text()
                self._set_live_background(self.live_bg)
                self._update_frame_loop()

    # =========================
    # Overlay Handling
    # =========================
    def _hide_overlay_text(self):
        try:
            self.video_text.place_forget()
        except Exception:
            pass

    def _show_overlay_text(self, text: str):
        self.video_text.configure(text=text, text_color=("black", "black"))
        self.video_text.place(relx=0.5, rely=0.5, anchor="center")
        self.video_text.lift()

    def _clear_video_label(self):
        self._set_live_background(self.placeholder_bg)

        self._tk_img = None
        if self.video_label is not None:
            self.video_label.configure(image="")
            self.video_label.image = None
        if self.video_text is not None:
            self._show_overlay_text("Kein Bild vorhanden\n(bitte Kamera auswählen)")
        self.update_idletasks()

    # =========================
    # Kamera Start/Stop
    # =========================
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
            self.current_cam_index = None
            self.is_streaming = False
            self._update_capture_button_state()
            return

        self.current_cam_index = index
        self.is_streaming = True

        self._set_live_background(self.live_bg)
        self._hide_overlay_text()

        self._update_capture_button_state()
        self._update_frame_loop()

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

    # =========================
    # Resize helper: FILL (Panel füllen, ggf. Crop)
    # =========================
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
        return resized[y0:y0 + target_h, x0:x0 + target_w]

    # =========================
    # Helper: RGB Bild im Panel anzeigen
    # =========================
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
    # Live Loop
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

    # =========================
    # Snapshot (speichert in captures/)
    # =========================
    def _save_snapshot(self):
        if self.current_cam_index is None or not str(self.camera_var.get()).startswith("Camera "):
            self._set_status("Bitte zuerst eine Kamera auswählen")
            return

        if self.last_frame is None:
            self._set_status("Kein Frame vorhanden (Kamera läuft? / Bild importiert?)")
            return

        ts = datetime.now().strftime("%d%m%Y_%H%M%S_%f")[:-3]
        cam = f"cam{self.current_cam_index}" if self.current_cam_index is not None else "img"
        filename = f"snapshot_{cam}_{ts}.png"
        path = os.path.join(self.capture_dir, filename)

        ok = cv2.imwrite(path, self.last_frame)
        if ok:
            if self.freeze_after_capture_var.get() and self.cap is not None:
                self.is_streaming = False
                self._set_status(f"Gespeichert & Live pausiert: {path}")
            else:
                self._set_status(f"Gespeichert: {path}")
        else:
            self._set_status("Speichern fehlgeschlagen")

    # =========================
    # Bild importieren (aus captures/ starten)
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
    # Bild exportieren (Save-Dialog; startet in captures/)
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

    # =========================
    # Optional: Alles aus captures/ exportieren (kopieren)
    # =========================
    def _export_all_captures(self):
        target_dir = filedialog.askdirectory(
            title="Zielordner wählen (alle Captures exportieren)",
            initialdir=os.path.abspath(self.capture_dir),
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

    def _set_status(self, msg: str):
        if self.status_label is not None:
            self.status_label.configure(text=msg)

    # =========================
    # Metadaten Popup
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
    # GUI bauen
    # =========================
    def _build_gui(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # LINKS viel Platz, RECHTS Mindestbreite
        main_frame.grid_columnconfigure(0, weight=6)
        main_frame.grid_columnconfigure(1, weight=0, minsize=360)
        main_frame.grid_rowconfigure(1, weight=1)

        # =========================
        # Oben Controls
        # =========================
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))

        camera_values = self._get_camera_menu_values()
        self._last_camera_values = camera_values

        self.camera_select = ctk.CTkOptionMenu(
            controls_frame,
            values=camera_values,
            variable=self.camera_var,
            command=self._on_camera_selected
        )
        self.camera_select.grid(row=0, column=0, padx=5)

        self.freeze_checkbox = ctk.CTkCheckBox(
            controls_frame,
            text="Nach Aufnahme pausieren",
            variable=self.freeze_after_capture_var,
            command=self._on_freeze_toggle
        )
        self.freeze_checkbox.grid(row=0, column=1, padx=10)

        ctk.CTkButton(controls_frame, text="Analyse starten").grid(row=0, column=2, padx=5)

        self.status_label = ctk.CTkLabel(controls_frame, text="", width=400, anchor="w")
        self.status_label.grid(row=0, column=3, padx=10)

        # =========================
        # Mitte Live-View
        # =========================
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
            relief="flat"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.video_text = ctk.CTkLabel(
            live_view,
            text="Kein Bild vorhanden\n(bitte Kamera auswählen)",
            font=ctk.CTkFont(size=18),
            fg_color="transparent"
        )
        self.video_text.place(relx=0.5, rely=0.5, anchor="center")

        # =========================
        # Unten Controls
        # =========================
        bottom_controls = ctk.CTkFrame(main_frame)
        bottom_controls.grid(row=2, column=0, sticky="w", pady=10)

        ctk.CTkButton(bottom_controls, text="Bild importieren", command=self._import_image).grid(row=0, column=0, padx=5)
        ctk.CTkButton(bottom_controls, text="Bild exportieren", command=self._export_image).grid(row=0, column=1, padx=5)

        self.capture_btn = ctk.CTkButton(bottom_controls, text="Bild aufnehmen", command=self._save_snapshot)
        self.capture_btn.grid(row=0, column=2, padx=5)

        ctk.CTkButton(bottom_controls, text="Alle Captures exportieren", command=self._export_all_captures).grid(row=0, column=3, padx=5)

        self._update_capture_button_state()

        # =========================
        # Rechts Panel
        # =========================
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
            command=lambda: self._open_metadata_popup(analysis_label)
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

        ctk.CTkButton(function_frame, text="Validieren", width=80).pack(side="left", padx=5, pady=10)
        ctk.CTkButton(function_frame, text="AutoAdjust", width=80).pack(side="left", padx=5, pady=10)

    # =========================
    # Sauber schließen
    # =========================
    def _on_close(self):
        # Pending refresh canceln
        if self._wm_refresh_after_id is not None:
            try:
                self.after_cancel(self._wm_refresh_after_id)
            except Exception:
                pass
            self._wm_refresh_after_id = None

        # Windows Hook entfernen
        self._uninstall_device_change_hook()

        self._stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()

