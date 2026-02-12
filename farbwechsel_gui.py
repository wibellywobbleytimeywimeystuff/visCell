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

from random import choice
import customtkinter as ctk
import numpy as np
import cv2

# ===== Farbschenma: Grundeinstellungen =====
appearance = "dark"
colormode = "Hellmodus"
ctk.set_appearance_mode(appearance)
ctk.set_default_color_theme("blue")


# =========================
# # 2 ) Hauptschleife
# =========================
class App(ctk.CTk):

    # =========================
    # 3 ) Grundeinstellungen
    # =========================
    def __init__(self):
        super().__init__()

        # ===== GUI: Fenster erstellen =====
        self.title("visCell")
        self.geometry("1440x980")

        self._panel_fg = "#BDBDBD"
        self._text_fg = "black"

        # ===== Widget-Referenzen =====
        self.live_view = None
        self.live_label = None
        self.analysis_frame = None
        self.analysis_label = None
        self.darkmode_btn = None

        self._build_footer()  # Button + GUI darstellen

    # =========================
    # 4 ) Farbschema: Wechselfunktion
    # =========================
    def _switchcolor(self):
        global appearance
        global colormode

        # ===== Farbschema: wechseln =====
        match appearance:
            case "light":
                appearance = "dark"
                colormode = "Hellmodus"
            case "dark":
                appearance = "light"
                colormode = "Dunkelmodus"

        if self.darkmode_btn is not None:
            self.darkmode_btn.configure(text=colormode)

        ctk.set_appearance_mode(appearance)  # Farbschema aktualisieren

        # ===== Textfarben aktualisieren =====
        if self.live_label is not None:
            self.live_label.configure(text_color=self._text_fg)
        if self.analysis_label is not None:
            self.analysis_label.configure(text_color=self._text_fg)

        # ===== Panelfarben aktualisieren =====
        if self.live_view is not None:
            self.live_view.configure(fg_color=self._panel_fg)
        if self.analysis_frame is not None:
            self.analysis_frame.configure(fg_color=self._panel_fg)

    # =========================
    # 5) GUI-Darstellung
    # =========================
    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=30)
        footer.pack(fill="x", side="bottom")

        # ===== Button Farbwechsel =====
        self.darkmode_btn = ctk.CTkButton(
            footer,
            text=colormode,
            command=self._switchcolor,
            width=110,
        )
        self.darkmode_btn.pack(side="right", padx=5, pady=5)

        # ===== Validierungsfunktion =====
        def daily_validation():
            pass

        # =========================
        # 6 ) Metadaten ändern Popup
        # =========================
        def open_popup():
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

            # ===== Eintragen der Metadaten =====
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

                if self.analysis_label is not None:
                    self.analysis_label.configure(
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

        # ===== GUI: Hauptbereich =====
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # ===== Steuerung: oben =====
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))

        camera_select = ctk.CTkOptionMenu(
            controls_frame, values=["Camera 0", "Camera 1", "Camera 2"]
        )
        camera_select.grid(row=0, column=0, padx=5)

        start_analysis_btn = ctk.CTkButton(controls_frame, text="Analyse starten")
        start_analysis_btn.grid(row=0, column=1, padx=5)

        # ===== GUI: Live-View =====
        self.live_view = ctk.CTkFrame(
            main_frame, fg_color=self._panel_fg, corner_radius=8
        )
        self.live_view.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        self.live_label = ctk.CTkLabel(
            self.live_view,
            text="Mikroskop-Aufnahme",
            font=ctk.CTkFont(size=18),
            text_color=self._text_fg,
        )
        self.live_label.place(relx=0.5, rely=0.5, anchor="center")

        # ===== Steuerung: unten =====
        bottom_controls = ctk.CTkFrame(main_frame)
        bottom_controls.grid(row=2, column=0, sticky="w", pady=10)

        ctk.CTkButton(bottom_controls, text="Bild importieren").grid(
            row=0, column=0, padx=5
        )
        ctk.CTkButton(bottom_controls, text="Bild exportieren").grid(
            row=0, column=1, padx=5
        )
        ctk.CTkButton(bottom_controls, text="Bild aufnehmen").grid(
            row=0, column=2, padx=5
        )

        # ===== GUI: rechts =====
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=3, sticky="nsew")
        right_panel.grid_rowconfigure(1, weight=1)

        # ===== Analyseergebnis =====
        self.analysis_frame = ctk.CTkFrame(
            right_panel, fg_color=self._panel_fg, corner_radius=8, height=200
        )
        self.analysis_frame.pack(fill="x", padx=10, pady=10)

        self.analysis_label = ctk.CTkLabel(
            self.analysis_frame,
            text="Analyseergebnis\n\nDatum:  \nPrüfer:  \nLabor: \n\nMikroskop:   \nVergrößerung:   \nProbe Nummer:   \n\nErythrozyten Anzahl:   \nLeukozyten Anzahl:   \nHefezellen Anzahl: ",
            justify="left",
            text_color=self._text_fg,
        )
        self.analysis_label.pack(side="left", padx=10, pady=10)

        # ===== Button: Exportieren =====
        export_frame = ctk.CTkFrame(right_panel, fg_color="transparent", height=200)
        export_frame.pack(fill="x", padx=10, pady=(0, 10))

        export_report_btn = ctk.CTkButton(
            export_frame, text="Metadaten ändern", command=open_popup
        )
        export_report_btn.pack(side="left", pady=10)

        export_report_btn = ctk.CTkButton(export_frame, text="Bericht exportieren")
        export_report_btn.pack(side="left", padx=10, pady=10)

        # ===== Schieberegler =====
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

        # ===== Helligkeit =====
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

        # ===== GUI: unter Schieberegler =====
        function_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        function_frame.pack(fill="x", padx=10, pady=10)

        validate_btn = ctk.CTkButton(function_frame, text="Validieren", width=80)
        validate_btn.pack(side="left", padx=5, pady=10)

        autoadjust_btn = ctk.CTkButton(function_frame, text="AutoAdjust", width=80)
        autoadjust_btn.pack(side="left", padx=5, pady=10)


# =========================
# 7 ) Start der Anwendung
# =========================
if __name__ == "__main__":
    app = App()
    app.mainloop()
