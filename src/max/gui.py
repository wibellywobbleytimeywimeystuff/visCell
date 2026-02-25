"""
Modul: guiuiuiuiuiui.py
Beschreibung:
Dieses Skript ist die grafische Benutzeroberflächedes Projekts Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.

Autor: Marlon Aust
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =============== Bibliotheken ===============
import customtkinter as ctk
import numpy as np
import cv2
from datetime import date

# =============== Farbdarstellung ===============
appearance = "dark"
colormode = "Hellmodus"
ctk.set_appearance_mode(appearance)
ctk.set_default_color_theme("blue")

# =============== Grundeinstellungen ===============
current_date = date.today()
current_date = current_date.strftime("%d.%m.%Y")

last_entry = {
    "name": "Martin Faust",
    "labor_name": "Kühle Sache GmbH",
    "equiptment": "Swift SW350B",
    "magnification": "400fach",
    "probe_number": "#017042027",
    "notes": "Feld für Notizen",
}
ai_result_ery = "/"
ai_result_leuco = "/"
ai_result_yeast = "/"


# =============== Hauptschleife ===============
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # =============== GUI ===============
        self.title("PLATZHALTER")
        self.geometry("1400x800")
        self.minsize(1200, 700)

        # =============== Farben/Defaults ===============
        self._panel_fg = "#BDBDBD"
        self._text_fg = "black"
        self._wrap_len = 250

        # =============== GUI Variablen ===============
        self.current_date = current_date
        self.last_entry = last_entry
        self.ai_result_ery = ai_result_ery
        self.ai_result_leuco = ai_result_leuco
        self.ai_result_yeast = ai_result_yeast

        # =============== Widget Referenzen ===============
        self.live_view = None
        self.live_label = None
        self.analysis_frame = None
        self.analysis_label = None
        self.darkmode_btn = None

        # =============== GUI Aufbau ===============
        self._build_footer()
        self._build_main()

    # =============== Hilfsfunktionen ===============
    def _make_analysis_text(self):
        return f"""\
Datum: {self.current_date}
Prüfer: {self.last_entry["name"]}
Labor: {self.last_entry["labor_name"]}

Mikroskop: {self.last_entry["equiptment"]}
Vergrößerung: {self.last_entry["magnification"]}
Probe Nummer: {self.last_entry["probe_number"]}

Erythrozyten Anzahl: {self.ai_result_ery}
Leukozyten Anzahl: {self.ai_result_leuco}
Hefezellen Anzahl: {self.ai_result_yeast}

Notizen:
{self.last_entry["notes"]}
"""

    # =============== Farbe wechseln ===============
    def _switchcolor(self):
        global appearance
        global colormode

        match appearance:
            case "light":
                appearance = "dark"
                colormode = "Hellmodus"
            case "dark":
                appearance = "light"
                colormode = "Dunkelmodus"

        if self.darkmode_btn is not None:
            self.darkmode_btn.configure(text=colormode)

        ctk.set_appearance_mode(appearance)

        if self.live_label is not None:
            self.live_label.configure(text_color=self._text_fg)
        if self.analysis_label is not None:
            self.analysis_label.configure(text_color=self._text_fg)

        if self.live_view is not None:
            self.live_view.configure(fg_color=self._panel_fg)
        if self.analysis_frame is not None:
            self.analysis_frame.configure(fg_color=self._panel_fg)

    # =============== Metadaten ändern ===============
    def _open_popup(self):
        popup = ctk.CTkToplevel(self)
        popup.title("Metadaten bearbeiten")
        popup.geometry("400x400")
        popup.grab_set()

        title_label_popup = ctk.CTkLabel(
            popup,
            text="Metadaten bearbeiten",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title_label_popup.pack(pady=10)

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

        # =============== Metadaten Popup ===============
        for field in fields:
            frame = ctk.CTkFrame(popup)
            frame.pack(fill="x", padx=20, pady=5)

            label = ctk.CTkLabel(frame, text=field, width=100, anchor="w")
            label.pack(side="left")

            entry = ctk.CTkEntry(frame)
            entry.pack(side="right", fill="x", expand=True)

            entries[field] = entry

        # =============== Metadaten abgleichen ===============
        def save_metadata():
            self.last_entry["name"] = (
                entries["Prüfer"].get()
                if entries["Prüfer"].get() != ""
                else self.last_entry["name"]
            )

            self.last_entry["labor_name"] = (
                entries["Labor"].get()
                if entries["Labor"].get() != ""
                else self.last_entry["labor_name"]
            )

            self.last_entry["equiptment"] = (
                entries["Equipment"].get()
                if entries["Equipment"].get() != ""
                else self.last_entry["equiptment"]
            )

            self.last_entry["magnification"] = (
                entries["Vergrößerung"].get()
                if entries["Vergrößerung"].get() != ""
                else self.last_entry["magnification"]
            )

            self.last_entry["probe_number"] = (
                entries["Probennummer"].get()
                if entries["Probennummer"].get() != ""
                else self.last_entry["probe_number"]
            )

            self.last_entry["notes"] = (
                entries["Notizen"].get()
                if entries["Notizen"].get() != ""
                else self.last_entry["notes"]
            )

            analysedatum = entries["Analysedatum"].get() or self.current_date

            # =============== Metadaten übernehmen ===============
            if self.analysis_label is not None:
                self.analysis_label.configure(
                    text=(
                        f"Datum: {analysedatum}\n"
                        f"Prüfer: {self.last_entry['name']}\n"
                        f"Labor: {self.last_entry['labor_name']}\n\n"
                        f"Mikroskop: {self.last_entry['equiptment']}\n"
                        f"Vergrößerung: {self.last_entry['magnification']}\n"
                        f"Probe Nummer: {self.last_entry['probe_number']}\n\n"
                        f"Erythrozyten Anzahl: {self.ai_result_ery}\n"
                        f"Leukozyten Anzahl: {self.ai_result_leuco}\n"
                        f"Hefezellen Anzahl: {self.ai_result_yeast}\n\n"
                        "Notizen: \n"
                        f"{self.last_entry['notes']}\n"
                    ),
                )

            popup.destroy()

        save_btn = ctk.CTkButton(popup, text="Speichern", command=save_metadata)
        save_btn.pack(pady=20)

    # =============== Footer ===============
    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=30)
        footer.pack(fill="x", side="bottom")

        version_label = ctk.CTkLabel(footer, text="version x.xx")
        version_label.pack(side="left", padx=10)

        self.darkmode_btn = ctk.CTkButton(
            footer,
            text=colormode,
            command=self._switchcolor,
            width=80,
        )
        self.darkmode_btn.pack(side="right", padx=5)

    # =============== Hauptbereich ===============
    def _build_main(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        self._build_controls(main_frame)
        self._build_live_view(main_frame)
        self._build_bottom_controls(main_frame)
        self._build_right_panel(main_frame)

    # =============== Kameraauswahl ===============
    def _build_controls(self, main_frame):
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))

        camera_select = ctk.CTkOptionMenu(
            controls_frame,
            values=["Camera 0", "Camera 1", "Camera 2"],
        )
        camera_select.grid(row=0, column=0, padx=5)

        start_analysis_btn = ctk.CTkButton(controls_frame, text="Analyse durchführen")
        start_analysis_btn.grid(row=0, column=1, padx=5)

    # =============== LIVE-VIEW ===============
    def _build_live_view(self, main_frame):
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

    # =============== Bild Buttons ===============
    def _build_bottom_controls(self, main_frame):
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

    # =============== Auswertungspanel ===============
    def _build_right_panel(self, main_frame):
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=3, sticky="nsew")
        right_panel.grid_rowconfigure(1, weight=1)

        self._build_analysis_panel(right_panel)
        self._build_export_buttons(right_panel)
        self._build_sliders(right_panel)
        self._build_function_buttons(right_panel)

    # =============== Analyseergebnis ===============
    def _build_analysis_panel(self, right_panel):
        self.analysis_frame = ctk.CTkFrame(
            right_panel,
            fg_color=self._panel_fg,
            corner_radius=8,
            height=200,
        )
        self.analysis_frame.pack(fill="x", padx=10, pady=10)

        title_label = ctk.CTkLabel(
            self.analysis_frame,
            text="        Auswertung der Analyse",
            font=("Arial", 12, "bold"),
            text_color=self._text_fg,
            justify="left",
        )
        title_label.pack(anchor="w", padx=10, pady=(10, 4))

        self.analysis_label = ctk.CTkLabel(
            self.analysis_frame,
            text=self._make_analysis_text(),
            justify="left",
            text_color=self._text_fg,
            wraplength=self._wrap_len,  # overflow
        )
        self.analysis_label.pack(
            anchor="w", padx=10, pady=(0, 10)
        )  # 35 für gleiche x-dis

    # =============== Metadaten Buttons ===============
    def _build_export_buttons(self, right_panel):
        export_frame = ctk.CTkFrame(right_panel, fg_color="transparent", height=200)
        export_frame.pack(fill="x", padx=10, pady=(0, 10))

        edit_metadata_btn = ctk.CTkButton(
            export_frame,
            text="Metadaten ändern",
            command=self._open_popup,
        )
        edit_metadata_btn.pack(side="left", pady=10)

        generate_report_btn = ctk.CTkButton(export_frame, text="Bericht generieren")
        generate_report_btn.pack(side="left", padx=10, pady=10)

    # =============== Schieberegler ===============
    def _build_sliders(self, right_panel):
        slider_frame = ctk.CTkFrame(right_panel)
        slider_frame.pack(fill="x", padx=10, pady=10)

        brightness_label = ctk.CTkLabel(slider_frame, text="Helligkeit")
        brightness_label.pack(anchor="w", padx=10, pady=10)

        brightness_slider = ctk.CTkSlider(slider_frame, from_=0, to=100)
        brightness_slider.pack(fill="x", pady=(0, 15))

        saturation_label = ctk.CTkLabel(slider_frame, text="Sättigung")
        saturation_label.pack(anchor="w", padx=10, pady=5)

        saturation_slider = ctk.CTkSlider(slider_frame, from_=0, to=100)
        saturation_slider.pack(fill="x", pady=10)

    # =============== Validieren/Schnellanpassung ===============
    def _build_function_buttons(self, right_panel):
        function_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        function_frame.pack(fill="x", padx=10, pady=10)

        validate_btn = ctk.CTkButton(function_frame, text="Validieren", width=80)
        validate_btn.pack(side="left", padx=5, pady=10)

        autoadjust_btn = ctk.CTkButton(
            function_frame, text="Schnellanpassung", width=80
        )
        autoadjust_btn.pack(side="left", padx=5, pady=10)


if __name__ == "__main__":
    app = App()
    app.mainloop()
