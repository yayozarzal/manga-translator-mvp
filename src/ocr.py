from typing import List, Dict, Optional

from PIL import Image
from manga_ocr import MangaOcr


class MangaTextOCR:
    """
    OCR especializado para texto japonés en manga.
    Usa Manga OCR para reconocer texto dentro de globos recortados.
    """

    def __init__(self):
        self.model: Optional[MangaOcr] = None

    def get_model(self) -> MangaOcr:
        """
        Carga el modelo OCR una sola vez.
        """
        if self.model is None:
            self.model = MangaOcr()
        return self.model

    def read_image(self, image: Image.Image) -> str:
        """
        Extrae texto de una imagen PIL.
        """
        model = self.get_model()

        image_rgb = image.convert("RGB")
        text = model(image_rgb)

        return text.strip()

    def read_bubbles(self, cropped_bubbles: List[Dict]) -> List[Dict]:
        """
        Aplica OCR a cada globo. Un fallo individual no detiene el proceso.
        """
        results = []

        for bubble in cropped_bubbles:
            detected_text = ""
            ocr_ok = False
            ocr_error = None

            try:
                detected_text = self.read_image(bubble["crop"]).strip()

                if detected_text:
                    ocr_ok = True
                else:
                    ocr_error = "No se reconoció texto en este globo."

            except Exception as error:
                ocr_error = str(error)

            results.append({
                **bubble,
                "text": detected_text,
                "ocr_ok": ocr_ok,
                "ocr_error": ocr_error
            })

        return results