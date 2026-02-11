"""
Modul: farbwechsel.py
Beschreibung:
Dieses Skript ändert dynamisch die Hintergrundfarbe der
grafischen Benutzeroberfläche von Dunkel auf Hell und zurück.

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# =========================
# 1 ) Einbinden der Bibliothek
# =========================
import customtkinter as ctk


# =========================
# 2 ) Hauptschleife
# =========================
class App(ctk.CTk):
    # =========================
    # 3 ) Grundeinstellungen
    # =========================
    def __init__(self):
        super().__init__()

        ctk.set_default_color_theme("blue")
        self.title("Darkmode/lightmode")
        self.geometry("600x400")

        self.appearance = "dark"
        self.colormode_text = "Hellmodus"
        ctk.set_appearance_mode(self.appearance)

        self._build_footer()  # Button darstellen

    # =========================
    # 3 ) Farbwechsel
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
    # 4 ) GUI-Darstellung
    # =========================
    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=30)
        footer.pack(fill="x", side="bottom")

        # ===== Button Farbwechsel =====
        self.darkmode_btn = ctk.CTkButton(
            footer,
            text=self.colormode_text,
            command=self._switchcolor,
            width=110,
        )
        self.darkmode_btn.pack(side="right", padx=5, pady=5)


# =========================
# 5 ) Start der Anwendung
# =========================
if __name__ == "__main__":
    app = App()
    app.mainloop()
