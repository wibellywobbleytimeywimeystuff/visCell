import customtkinter as ctk
from tkinter import filedialog
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# ==============================
# 1 ) DATENSTRUKTUR
# ==============================
@dataclass
class Params:
    brightness: int       # Helligkeit (Offset, wird addiert)
    saturation: float     # Sättigung (Multiplikator im HSV-S Kanal)


# ==============================
# 2 ) BILDBEARBEITUNG
# ==============================
def apply_brightness(img: np.ndarray, brightness: int) -> np.ndarray:
    """
    Helligkeit:
    OpenCV nutzt konzeptionell: neues_pixel = 1.0 * altes_pixel + brightness
    -> brightness (beta) ist ein Offset, der auf jeden Pixel addiert wird.
    """
    return cv2.convertScaleAbs(img, alpha=1.0, beta=brightness)


def apply_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """
    Sättigung:
    - Bild in HSV umwandeln
    - S-Kanal (Index 1) multiplizieren
    - auf 0..255 begrenzen
    - zurück nach BGR
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process_image(img: np.ndarray, params: Params) -> np.ndarray:
    """
    Pipeline: Helligkeit -> Sättigung
    """
    img = apply_brightness(img, params.brightness)
    img = apply_saturation(img, params.saturation)
    return img


# ==============================
# 3 ) GUI
# ==============================
class ImageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Automatische Bildanpassung (CustomTkinter)")
        self.geometry("1100x600")

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Bilddaten
        self.original_img = None
        self.preview_img = None
        self.ctk_image = None

        # --------------------------
        # LINKER BEREICH (Steuerung)
        # --------------------------
        self.control_frame = ctk.CTkFrame(self, width=320)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.control_frame.pack_propagate(False)

        ctk.CTkLabel(self.control_frame, text="Steuerung", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=(5, 10), anchor="w"
        )

        ctk.CTkButton(
            self.control_frame,
            text="Bild laden",
            command=self.load_image
        ).pack(pady=(0, 15), fill="x")

        # Helligkeit
        ctk.CTkLabel(self.control_frame, text="Helligkeit").pack(anchor="w")
        self.brightness_var = ctk.IntVar(value=20)

        ctk.CTkSlider(
            self.control_frame,
            from_=-100,
            to=100,
            variable=self.brightness_var,
            command=self.on_slider_change
        ).pack(fill="x", pady=(5, 15))

        # Sättigung
        ctk.CTkLabel(self.control_frame, text="Sättigung").pack(anchor="w")
        self.saturation_var = ctk.DoubleVar(value=1.2)

        ctk.CTkSlider(
            self.control_frame,
            from_=0.0,
            to=2.0,
            variable=self.saturation_var,
            command=self.on_slider_change
        ).pack(fill="x", pady=(5, 10))

        # Live-Anzeige der Werte
        self.params_label = ctk.CTkLabel(self.control_frame, text="Aktuell: Helligkeit = 39 | Sättigung = 1.27")
        self.params_label.pack(pady=(0, 15), anchor="w")

        # Auto-Optimierung
        ctk.CTkButton(
            self.control_frame,
            text="Auto-Optimierung",
            command=self.apply_auto
        ).pack(pady=(0, 10), fill="x")

        # Nullung
        ctk.CTkButton(
            self.control_frame,
            text="Original",
            command=self.apply_nullung
        ).pack(pady=(0,10), fill="x")



        # --------------------------
        # 4 ) RECHTER BEREICH (Bild)
        # --------------------------
        self.image_label = ctk.CTkLabel(self, text="Kein Bild geladen")
        self.image_label.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    # ==========================
    # 5 ) GUI Aktionen
    # ==========================
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            self.image_label.configure(text="Bild konnte nicht geladen werden.", image=None)
            return

        self.original_img = img

        
        self.apply_auto()

    def apply_auto(self):
        self.brightness_var.set(39)    
        self.saturation_var.set(1.27)   


        self.update_preview()
    
    def apply_nullung(self):
        self.brightness_var.set(0)
        self.saturation_var.set(1.0)

        self.update_preview()
        

    def on_slider_change(self, _=None):
        self.update_preview()

    def update_params_label(self):
        b = int(self.brightness_var.get())
        s = float(self.saturation_var.get())
        self.params_label.configure(text=f"Aktuell: Helligkeit = {b} | Sättigung = {s:.2f}")

    def update_preview(self):
        if self.original_img is None:
            return

        
        self.update_params_label()

        params = Params(
            brightness=int(self.brightness_var.get()),
            saturation=float(self.saturation_var.get())
        )

        self.preview_img = process_image(self.original_img, params)
        self.show_image(self.preview_img)

    def show_image(self, img: np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        max_w, max_h = 700, 500
        scale = min(max_w / w, max_h / h, 1.0)

        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(rgb)

        # CustomTkinter Image muss gespeichert werden (sonst verschwindet es)
        self.ctk_image = ctk.CTkImage(light_image=pil_img, size=pil_img.size)
        self.image_label.configure(image=self.ctk_image, text="")


# ==============================
# START
# ==============================
if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()
