"""
Modul: validation_gui.py
Beschreibung:
Zur Validierung werden DUMMY-Werte simuliert und mit einer Toleranz von 1% (0,01) verglichen

Autor: Marlon Aust
Projektname: visCell
Projekt: Entwicklung einer portablen Windows-Anwendung zur automatisierten KI-Analyse
von mikroskopischen Zellstrukturen.
"""

# ============================
# 1 ) Bibliotheken importieren
# ============================
import threading
import time
from tkinter import messagebox
import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

deltatol = 0.01  # Toleranz (dezimal)


# =========================
# # 2 ) Hauptschleife
# =========================
class ValidierungGUI(ctk.CTk):

    # =========================
    # 3 ) Grundeinstellungen
    # =========================
    def __init__(self):
        super().__init__()

        # ===== GUI: Fenster erstellen =====
        self.title("Validierung")
        self.geometry("420x220")

        # ===== DUMMY-Analyse =====
        self.soll_ery = 100  # Referenzwert (Soll)
        self.ki_ery = 101  # Dummy-KI-Ausgabe
        self.validierung_ok = False  # Validierung ok/nok

        # =========================
        # 4 ) GUI-Darstellung
        # =========================
        self.headline = ctk.CTkLabel(self, text="Tägliche Validierung", font=ctk.CTkFont(size=18, weight="bold"))
        self.headline.pack(pady=(18, 8))

        # ===== Status: Label =====
        self.status_label = ctk.CTkLabel(self, text="Bereit", text_color="#00B7FF")
        self.status_label.pack(pady=(0, 8))

        # ===== GUI: Fortschrittsbalken =====
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.pack(pady=6)
        self.progress_frame.pack_forget()  # erst ab Augmentation sichtbar

        # ===== Fortschrittsbalken: Design =====
        self.progress = ctk.CTkProgressBar(self.progress_frame, width=300, height=16)
        self.progress.pack(pady=6)
        self.progress.set(0)

        self.progress_text = ctk.CTkLabel(self.progress_frame, text="0 %", font=ctk.CTkFont(size=16, weight="bold"),)
        self.progress_text.pack()

        # ===== Button: Validieren =====
        self.btn_validate = ctk.CTkButton(self, text="Validieren", command=self.start_validation)
        self.btn_validate.pack(pady=(10, 6))

    # =========================
    # 5 ) Validierung
    # =========================
    # ===== Validierung: starten =====
    def start_validation(self):
        self.btn_validate.configure(state="disabled")  # Button sperren

        self._show_progress()  # Progress anzeigen
        thread = threading.Thread(target=self._validation_process, daemon=True)  # Hintergrundprozess
        thread.start()

    # ===== Validierungsprozess (5 steps) =====
    # Delays sorgen für einen professionelleren look
    def _validation_process(self):
        time.sleep(0.4)

        # Referenz laden
        self._status("Lade Referenzdaten...", "#00B7FF")
        self._update_progress(1, 5)  # 1/5
        time.sleep(0.1)

        soll = int(self.soll_ery)  # Referenzbild

        # KI: analysiert
        self._status("KI analysiert Referenzbild...", "#00B7FF")
        self._update_progress(2, 5)  # 2/5
        time.sleep(0.8)

        ist = int(self.ki_ery)  # KI holen

        # KI: in Toleranzbereich?
        self._status("Prüfe Ergebnis (Regel A)...", "#00B7FF")
        self._update_progress(3, 5)  # 3/5
        time.sleep(0.05)

        ok, tol = self._regel_check(ist=ist, soll=soll)
        self.validierung_ok = bool(ok)  # KI in Toleranzbereich!

        # Ergebnis anzeigen
        self._status("Ergebnis bereit (Bediener bestätigen)...", "#00B7FF")
        self._update_progress(4, 5)  # 4/5
        time.sleep(0.6)

        # PopUp: Validierung ok?
        self.after(0, lambda: self._popup_confirm(ist, soll, tol))
        self._update_progress(5, 5)  # 5/5

    # =========================
    # 6 ) Toleranz
    # =========================
    # ===== Toleranz: Grenzen berechnen =====
    def _regel_check(self, ist: int, soll: int) -> tuple[bool, int]:
        toleranz = max(1, round(soll * deltatol))
        untergrenze = soll - toleranz
        obergrenze = soll + toleranz

        ok = untergrenze <= ist <= obergrenze
        return ok, toleranz

    # =========================
    # 7 ) Validierung bestätigen
    # =========================
    # ===== Validierung: Pop-Up bestätigen =====
    def _popup_confirm(self, ist: int, soll: int, tol: int):

        if self.validierung_ok:  # Validierung ok
            text = (
                f"Validierung erfolgreich.\n\n"
                f"Referenzwert: {soll}\n"
                f"Istwert: {ist}\n"
                f"Toleranz: ±{tol}%\n\n"
                f"Bestätigen Sie die Validierung?")
        else:  # Validierung nok
            text = (
                f"Validierung nicht erfolgreich.\n\n"
                f"Referenzwert: {soll}\n"
                f"Istwert: {ist}\n"
                f"Toleranz: ±{tol}%\n\n"
                f"Trotzdem als erfolgreich bestätigen?")

        confirmed = messagebox.askyesno("Validierung", text)

        # ===== Status: setzen =====
        if confirmed and self.validierung_ok:
            self._status("Validierung bestätigt ✅", "#00FF66")
        elif confirmed and not self.validierung_ok:
            self._status("Bestätigt, aber NICHT bestanden ⚠️", "#FFAA00")
        else:
            self._status("Nicht bestätigt ❌", "#FF4444")

        self.btn_validate.configure(state="normal")  # Button freigeben

    # ===== Status: aktualisieren =====
    def _status(self, text: str, farbe: str):
        self.after(0, lambda: self.status_label.configure(text=text, text_color=farbe))

    # =========================
    # 8 ) Fortschrittsbalken füllen
    # =========================
    # count = aktuelle Phase der Validierung
    # total = maximale Anzahl der Phasen (5)
    # count 2: 2/5 = 40 % | count 5: 5/5 = 100 %
    def _update_progress(self, count: int, total: int):
        total = max(1, int(total))  # Werte normalisiert
        percent = max(0, min(count / total, 1))  # % pro Phase

        # ===== GUI: aktualisieren =====
        def gui_update():
            self.progress.set(percent)  # Fortschritt
            self.progress_text.configure(text=f"{int(percent * 100)} %")  # Status

        self.after(0, gui_update)  # GUI update

    def _show_progress(self):
        self.progress.set(0)  # Startwert
        self.progress_text.configure(text="0 %")  # Default
        self.progress_frame.pack(pady=6)


# =========================
# 9 ) Start der Anwendung
# =========================
if __name__ == "__main__":
    app = ValidierungGUI()
    app.mainloop()
