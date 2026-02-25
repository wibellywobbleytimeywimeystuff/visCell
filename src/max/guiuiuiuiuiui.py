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

        # =============== Farbe wechseln ===============
        def switchcolor():
            global appearance
            global colormode

            match appearance:
                case "light":
                    appearance = "dark"
                    colormode = "Hellmodus"
                case "dark":
                    appearance = "light"
                    colormode = "Dunkelmodus"

            darkmode_btn.configure(text=colormode)
            ctk.set_appearance_mode(appearance)

            live_label.configure(text_color="black")
            analysis_label.configure(text_color="black")
            live_view.configure(fg_color="#BDBDBD")
            analysis_frame.configure(fg_color="#BDBDBD")

        # =============== Fußleiste ===============
        footer = ctk.CTkFrame(self, height=30)
        footer.pack(fill="x", side="bottom")

        version_label = ctk.CTkLabel(footer, text="version x.xx")
        version_label.pack(side="left", padx=10)

        darkmode_btn = ctk.CTkButton(
            footer,
            text=colormode,
            command=switchcolor,
            width=80,
        )
        darkmode_btn.pack(side="right", padx=5)

        # =============== Metadaten ändern ===============
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
                last_entry["name"] = (
                    entries["Prüfer"].get()
                    if entries["Prüfer"].get() != ""
                    else last_entry["name"]
                )

                last_entry["labor_name"] = (
                    entries["Labor"].get()
                    if entries["Labor"].get() != ""
                    else last_entry["labor_name"]
                )

                last_entry["equiptment"] = (
                    entries["Equipment"].get()
                    if entries["Equipment"].get() != ""
                    else last_entry["equiptment"]
                )

                last_entry["magnification"] = (
                    entries["Vergrößerung"].get()
                    if entries["Vergrößerung"].get() != ""
                    else last_entry["magnification"]
                )

                last_entry["probe_number"] = (
                    entries["Probennummer"].get()
                    if entries["Probennummer"].get() != ""
                    else last_entry["probe_number"]
                )

                last_entry["notes"] = (
                    entries["Notizen"].get()
                    if entries["Notizen"].get() != ""
                    else last_entry["notes"]
                )
                # =============== Metadaten übernehmen ===============
                analysis_label.configure(
                    text=(
                        f"Datum: {entries['Analysedatum'].get() or current_date}\n"
                        f"Prüfer: {last_entry['name']}\n"
                        f"Labor: {last_entry['labor_name']}\n\n"
                        f"Mikroskop: {last_entry['equiptment']}\n"
                        f"Vergrößerung: {last_entry['magnification']}\n"
                        f"Probe Nummer: {last_entry['probe_number']}\n\n"
                        f"Erythrozyten Anzahl: {ai_result_ery}\n"
                        f"Leukozyten Anzahl: {ai_result_leuco}\n"
                        f"Hefezellen Anzahl: {ai_result_yeast}\n\n"
                        "Notizen: \n"
                        f"{last_entry['notes']}"
                    ),
                )

                popup.destroy()

            save_btn = ctk.CTkButton(popup, text="Speichern", command=save_metadata)
            save_btn.pack(pady=20)

        # =============== Hauptbereich ===============
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # =============== Kameraauswahl ===============
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
        live_view = ctk.CTkFrame(main_frame, fg_color="#BDBDBD", corner_radius=8)
        live_view.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        live_label = ctk.CTkLabel(
            live_view,
            text="Mikroskop-Aufnahme",
            font=ctk.CTkFont(size=18),
            text_color="black",
        )
        live_label.place(relx=0.5, rely=0.5, anchor="center")

        # =============== Bild Buttons ===============
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
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=3, sticky="nsew")

        right_panel.grid_rowconfigure(1, weight=1)

        # =============== Analyseergebnis ===============
        analysis_frame = ctk.CTkFrame(
            right_panel,
            fg_color="#BDBDBD",
            corner_radius=8,
            height=200,
        )
        analysis_frame.pack(fill="x", padx=10, pady=10)

        analysis_text = f"""\
Datum: {current_date}
Prüfer: {last_entry["name"]}
Labor: {last_entry["labor_name"]}

Mikroskop: {last_entry["equiptment"]}
Vergrößerung: {last_entry["magnification"]}
Probe Nummer: {last_entry["probe_number"]}

Erythrozyten Anzahl: {ai_result_ery}
Leukozyten Anzahl:   {ai_result_leuco}
Hefezellen Anzahl:   {ai_result_yeast}

Notizen:
{last_entry["notes"]}
"""

        title_label = ctk.CTkLabel(
            analysis_frame,
            text="        Auswertung der Analyse",
            font=("Arial", 12, "bold"),
            text_color="black",
            justify="left",
        )

        analysis_label = ctk.CTkLabel(
            analysis_frame,
            text=analysis_text,
            justify="left",
            text_color="black",
            wraplength=250,  # overflow
        )

        title_label.pack(anchor="w", padx=10, pady=(10, 4))
        analysis_label.pack(anchor="w", padx=10, pady=(0, 10))  # 35 für gleiche x-dis

        # =============== Metadaten Buttons ===============
        export_frame = ctk.CTkFrame(right_panel, fg_color="transparent", height=200)
        export_frame.pack(fill="x", padx=10, pady=(0, 10))

        export_report_btn = ctk.CTkButton(
            export_frame,
            text="Metadaten ändern",
            command=open_popup,
        )
        export_report_btn.pack(side="left", pady=10)

        export_report_btn = ctk.CTkButton(export_frame, text="Bericht generieren")
        export_report_btn.pack(side="left", padx=10, pady=10)

        # =============== Schieberegler ===============
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
