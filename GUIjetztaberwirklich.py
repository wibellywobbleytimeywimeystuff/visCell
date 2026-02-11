"""
Modul: gui.py
Beschreibung:
Dieses Skript ist die grafische Benutzeroberflächedes Projekts Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse von mikroskopischen Zellstrukturen.

Autor: Max Sielhorst
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""





# ============================
# 1 ) Bibliotheken importieren
# ============================


from random import choice
import customtkinter as ctk
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image, ImageTk


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_default_color_theme("blue")
        self.title("Darkmode/lightmode")
        self.geometry("1440x980")

        # ===== Appearance =====
        self.appearance = "dark"
        self.colormode_text = "Hellmodus"
        ctk.set_appearance_mode(self.appearance)

        # ===== Kamera / Video =====
        self.cap = None
        self.is_streaming = False
        self.current_cam_index = None
        self.last_frame = None
        self._tk_img = None  # Referenz für Tk-Image (wichtig!)

        # ===== Speicherordner für Snapshots =====
        self.capture_dir = "captures"
        os.makedirs(self.capture_dir, exist_ok=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_gui()
        self._build_footer()

    # =========================
    # Appearance Toggle
    # =========================
    def _switchcolor(self):
        if self.appearance == "light":
            self.colormode_text = "Hellmodus"
            self.appearance = "dark"
        else:
            self.colormode_text = "Dunkelmodus"
            self.appearance = "light"

        ctk.set_appearance_mode(self.appearance)
        self.darkmode_btn.configure(text=self.colormode_text)

    # =========================
    # Kameras finden
    # =========================
    def _detect_cameras(self, max_index: int = 5):
        found = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(f"Camera {i}")
            cap.release()
        return found  # kann leer sein

    # =========================
    # Dropdown Callback
    # =========================
    def _on_camera_selected(self, choice: str):
        if choice.startswith("Camera "):
            idx = int(choice.split()[-1])
            self._start_camera(idx)
        else:
            self._stop_camera()
            self.video_label.configure(text="Kein Bild vorhanden\n(bitte Kamera auswählen)", image=None)

    # =========================
    # Kamera starten/stoppen
    # =========================
    def _start_camera(self, index: int):
        self._stop_camera()

        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.video_label.configure(text=f"Kamera {index} konnte nicht geöffnet werden", image=None)
            self.cap = None
            return

        self.current_cam_index = index
        self.is_streaming = True
        self.video_label.configure(text="")  # Text weg
        self._update_frame_loop()

    def _stop_camera(self):
        self.is_streaming = False
        self.current_cam_index = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    # =========================
    # Live-Loop (Tk after) - robust
    # =========================
    def _update_frame_loop(self):
        if not self.is_streaming or self.cap is None:
            return

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame = frame

                # BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Live-View Größe (nach Wunsch anpassen)
                w, h = 900, 650
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

                pil_img = Image.fromarray(rgb)

                # Stabil für Tkinter: ImageTk.PhotoImage
                self._tk_img = ImageTk.PhotoImage(pil_img)
                self.video_label.configure(image=self._tk_img, text="")

        except Exception as e:
            print("Fehler im Live-Loop:", e)

        finally:
            self.after(33, self._update_frame_loop)

    # =========================
    # Snapshot: EIN Bild speichern (Button "Bild aufnehmen")
    # =========================
    def _save_snapshot(self):
        if self.last_frame is None:
            self._set_status("Kein Frame vorhanden (Kamera läuft?)")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cam = f"cam{self.current_cam_index}" if self.current_cam_index is not None else "camX"
        filename = f"snapshot_{cam}_{ts}.png"
        path = os.path.join(self.capture_dir, filename)

        ok = cv2.imwrite(path, self.last_frame)  # last_frame ist BGR -> OpenCV speichert korrekt
        if ok:
            self._set_status(f"Gespeichert: {path}")
        else:
            self._set_status("Speichern fehlgeschlagen")

    def _set_status(self, msg: str):
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
                    "Erythrozyten Anzahl: 200 Stück\n"
                    "Leukozyten Anzahl: 230 Stück\n"
                    "Hefezellen Anzahl: 2 Stück\n\n"
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

        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # =========================
        # Oben Controls
        # =========================
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))

        detected = self._detect_cameras(max_index=5)
        camera_values = ["Kamera auswählen..."] + detected
        if len(detected) == 0:
            camera_values = ["Kamera auswählen...", "Keine Kamera gefunden"]

        self.camera_select = ctk.CTkOptionMenu(
            controls_frame,
            values=camera_values,
            command=self._on_camera_selected
        )
        self.camera_select.grid(row=0, column=0, padx=5)
        self.camera_select.set("Kamera auswählen...")

        start_analysis_btn = ctk.CTkButton(controls_frame, text="Analyse starten")
        start_analysis_btn.grid(row=0, column=1, padx=5)

        self.status_label = ctk.CTkLabel(controls_frame, text="", width=500, anchor="w")
        self.status_label.grid(row=0, column=2, padx=10)

        # =========================
        # Mitte Live-View
        # =========================
        live_view = ctk.CTkFrame(main_frame, fg_color="#050505", corner_radius=8)
        live_view.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        self.video_label = ctk.CTkLabel(
            live_view,
            text="Kein Bild vorhanden\n(bitte Kamera auswählen)",
            font=ctk.CTkFont(size=18),
        )
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # =========================
        # Unten Controls
        # =========================
        bottom_controls = ctk.CTkFrame(main_frame)
        bottom_controls.grid(row=2, column=0, sticky="w", pady=10)

        ctk.CTkButton(bottom_controls, text="Bild importieren").grid(row=0, column=0, padx=5)
        ctk.CTkButton(bottom_controls, text="Bild exportieren").grid(row=0, column=1, padx=5)

        # WICHTIG: vorhandener Button speichert Snapshot aus Live-Feed
        ctk.CTkButton(bottom_controls, text="Bild aufnehmen", command=self._save_snapshot).grid(
            row=0, column=2, padx=5
        )

        # =========================
        # Rechts Panel
        # =========================
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=3, sticky="nsew")
        right_panel.grid_rowconfigure(1, weight=1)

        analysis_frame = ctk.CTkFrame(right_panel, fg_color="#000000", corner_radius=8, height=200)
        analysis_frame.pack(fill="x", padx=10, pady=10)

        analysis_label = ctk.CTkLabel(
            analysis_frame,
            text=(
                "ANALYSE_ERGEBNIS\n\n"
                "Datum:  \nPrüfer:  \nLabor: \n\n"
                "Mikroskop:   \nVergrößerung:   \nProbe Nummer:   \n\n"
                "Erythrozyten Anzahl:   \nLeukozyten Anzahl:   \nHefezellen Anzahl: "
            ),
            justify="left",
        )
        analysis_label.pack(side="left", padx=10, pady=10)

        export_frame = ctk.CTkFrame(right_panel, fg_color="transparent", height=200)
        export_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(
            export_frame,
            text="Metadaten ändern",
            command=lambda: self._open_metadata_popup(analysis_label)
        ).pack(side="left", pady=10)

        ctk.CTkButton(export_frame, text="Bericht exportieren").pack(side="left", padx=10, pady=10)

        function_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        function_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(function_frame, text="Validieren", width=80).pack(side="left", padx=5, pady=10)
        ctk.CTkButton(function_frame, text="AutoAdjust", width=80).pack(side="left", padx=5, pady=10)

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
    # Sauber schließen
    # =========================
    def _on_close(self):
        self._stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
