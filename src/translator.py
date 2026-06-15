from typing import List, Dict

from deep_translator import GoogleTranslator


class TextTranslator:
    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "es"
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        translator = GoogleTranslator(
            source=self.source_lang,
            target=self.target_lang
        )

        return translator.translate(text.strip())

    def translate_bubbles(self, ocr_results: List[Dict]) -> List[Dict]:
        translated_results = []

        for result in ocr_results:
            translated_text = ""
            translation_ok = False
            translation_error = None

            original_text = result.get("text", "").strip()
            ocr_ok = result.get("ocr_ok", bool(original_text))

            if not ocr_ok or not original_text:
                translation_error = (
                    "No se tradujo porque el OCR no produjo texto."
                )
            else:
                try:
                    translated_text = self.translate_text(original_text).strip()

                    if translated_text:
                        translation_ok = True
                    else:
                        translation_error = (
                            "El servicio devolvió una traducción vacía."
                        )

                except Exception as error:
                    translation_error = (
                        "No se pudo realizar la traducción. "
                        "Verifique la conexión a Internet. "
                        f"Detalle: {error}"
                    )

            translated_results.append({
                **result,
                "translated_text": translated_text,
                "translation_ok": translation_ok,
                "translation_error": translation_error
            })

        return translated_results