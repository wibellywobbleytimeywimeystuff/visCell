"""
Modul: gui.py
Beschreibung:
Dieses Skript ist die grafische Benutzeroberflächedes Projekts Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse von mikroskopischen Zellstrukturen.

Autor: Max Sielhorst
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""


# ===== Bibliotheken einbinden =====
from random import choice
import customtkinter as ctk
import numpy as np
import cv2


# ===== Farbdarstellung  =====
appearance = "dark"
ctk.set_appearance_mode(appearance)
ctk.set_default_color_theme("blue")

# ===== Hauptschleife =====
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ===== GUI =====
        self.geometry("1440x920")
        self.title("visCell")

        # =============== Unten ===============
        footer = ctk.CTkFrame(self, height=30)
        footer.pack(fill="x", side="bottom")

        # =============== Farbe ändern marlon ===============







        
        # =============== Hauptbereich ===============
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)


        # =============== Einstellungen oben ===============
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))

        camera_select = ctk.CTkOptionMenu(
            controls_frame, values=["Camera 0", "Camera 1", "Camera 2", "Live-View"]
        )
        camera_select.grid(row=0, column=0, padx=5)

app = App()
app.mainloop()