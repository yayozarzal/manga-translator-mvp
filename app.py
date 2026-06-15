import os
from typing import Optional

import gradio as gr
from PIL import Image

from src import editor
from src.image_utils import validate_image, normalize_image, get_image_info
from src.image_utils import (
    prepare_image,
    get_image_info
)
from src.detector import BubbleDetector
from src.ocr import MangaTextOCR
from src.translator import TextTranslator
from src.editor import BubbleEditor
from configs.settings import RANDOM_SEED
from src.reproducibility import set_reproducibility


set_reproducibility(RANDOM_SEED)


MODEL_PATH = "models/comic-speech-bubble-detector.pt"

# Si todavía no tienes un modelo entrenado, puedes usar temporalmente:
# MODEL_PATH = "yolov8n.pt"


detector: Optional[BubbleDetector] = None
ocr_engine = None
translator_engine = None
bubble_editor = None


def get_detector() -> BubbleDetector:
    """
    Carga el detector una sola vez.
    """
    global detector

    if detector is None:
        if not os.path.exists(MODEL_PATH) and MODEL_PATH != "yolov8n.pt":
            raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

        detector = BubbleDetector(
            model_path=MODEL_PATH,
            confidence_threshold=0.30
        )

    return detector


def load_manga_image(image: Image.Image):
    try:
        prepared_image, preparation_message = prepare_image(image)

        info = (
            f"{preparation_message}\n"
            f"{get_image_info(prepared_image)}"
        )

        return prepared_image, info

    except Exception as error:
        return None, f"Error al cargar la imagen: {error}"


def detect_bubbles(image: Image.Image):
    warnings = []

    try:
        image, preparation_message = prepare_image(image)
        warnings.append(preparation_message)

        detector = get_detector()
        annotated_image, detections = detector.detect(image)

        if not detections:
            return (
                annotated_image,
                "No se detectaron globos de diálogo.",
                [],
                "No se ejecutó OCR.",
                "No se ejecutó traducción.",
                [],
                [],
                image,
                (
                    "Procesamiento finalizado sin errores, "
                    "pero no se encontraron globos."
                )
            )

        cropped_bubbles = detector.crop_detections(
            image=image,
            detections=detections,
            padding=0,
            shrink_ratio=0.08
        )

        if not cropped_bubbles:
            return (
                annotated_image,
                "Se detectaron regiones, pero ninguna produjo un recorte válido.",
                [],
                "No se ejecutó OCR.",
                "No se ejecutó traducción.",
                [],
                [],
                image,
                "No se generaron recortes válidos."
            )

        ocr = get_ocr_engine()
        ocr_results = ocr.read_bubbles(cropped_bubbles)

        translator = get_translator_engine()
        translated_results = translator.translate_bubbles(ocr_results)

        editor = get_bubble_editor()
        cleaned_results = editor.clean_bubbles(translated_results)
        final_results = editor.insert_translations(cleaned_results)

        final_page = editor.compose_translated_page(
            original_image=image,
            final_results=final_results
        )

        gallery_items = []
        clean_gallery_items = []
        translated_gallery_items = []

        for result in final_results:
            bubble_id = result["id"]
            confidence = result["confidence"]

            gallery_items.append(
                (result["crop"], f"G{bubble_id} | {confidence}")
            )

            clean_gallery_items.append(
                (result["clean_crop"], f"G{bubble_id} limpio")
            )

            translated_gallery_items.append(
                (result["translated_crop"], f"G{bubble_id} traducido")
            )

            if not result.get("ocr_ok", False):
                warnings.append(
                    f"Globo {bubble_id}: "
                    f"{result.get('ocr_error', 'falló el OCR')}."
                )

            if not result.get("translation_ok", False):
                warnings.append(
                    f"Globo {bubble_id}: "
                    f"{result.get('translation_error', 'falló la traducción')}."
                )

        detection_text = "Detecciones encontradas:\n\n"

        for detection in detections:
            detection_text += (
                f"Globo {detection['id']}:\n"
                f"- Clase: {detection['class_name']}\n"
                f"- Confianza: {detection['confidence']}\n"
                f"- Coordenadas: "
                f"({detection['x1']}, {detection['y1']}) → "
                f"({detection['x2']}, {detection['y2']})\n"
                f"- Tamaño: "
                f"{detection['width']}x{detection['height']} px\n\n"
            )

        ocr_text = "Texto extraído por OCR:\n\n"

        for result in final_results:
            if result.get("ocr_ok"):
                content = result["text"]
            else:
                content = (
                    "[No se reconoció texto: "
                    f"{result.get('ocr_error', 'error desconocido')}]"
                )

            ocr_text += (
                f"Globo {result['id']}:\n"
                f"{content}\n\n"
            )

        translation_text = "Traducción automática al español:\n\n"

        for result in final_results:
            if result.get("translation_ok"):
                translation = result["translated_text"]
            else:
                translation = (
                    "[Sin traducción: "
                    f"{result.get('translation_error', 'error desconocido')}]"
                )

            translation_text += (
                f"Globo {result['id']}:\n"
                f"Original: {result.get('text', '')}\n"
                f"Traducción: {translation}\n\n"
            )

        successful_ocr = sum(
            1 for result in final_results
            if result.get("ocr_ok")
        )

        successful_translations = sum(
            1 for result in final_results
            if result.get("translation_ok")
        )

        status_text = (
            "Procesamiento terminado.\n"
            f"Globos detectados: {len(detections)}\n"
            f"Recortes válidos: {len(cropped_bubbles)}\n"
            f"OCR correctos: {successful_ocr}/{len(final_results)}\n"
            f"Traducciones correctas: "
            f"{successful_translations}/{len(final_results)}"
        )

        if warnings:
            status_text += "\n\nAdvertencias:\n- " + "\n- ".join(warnings)

        return (
            annotated_image,
            detection_text,
            gallery_items,
            ocr_text,
            translation_text,
            clean_gallery_items,
            translated_gallery_items,
            final_page,
            status_text
        )

    except FileNotFoundError as error:
        return (
            None,
            f"Error de modelo: {error}",
            [],
            "OCR no ejecutado.",
            "Traducción no ejecutada.",
            [],
            [],
            None,
            "No fue posible cargar el modelo."
        )

    except ValueError as error:
        return (
            None,
            f"Entrada inválida: {error}",
            [],
            "OCR no ejecutado.",
            "Traducción no ejecutada.",
            [],
            [],
            None,
            f"Error de validación: {error}"
        )

    except Exception as error:
        return (
            None,
            f"Error inesperado: {error}",
            [],
            "OCR no ejecutado.",
            "Traducción no ejecutada.",
            [],
            [],
            None,
            (
                "El procesamiento se detuvo por un error no controlado. "
                f"Detalle: {error}"
            )
        )
        
def get_ocr_engine() -> MangaTextOCR:
    """
    Carga el motor OCR una sola vez.
    """
    global ocr_engine

    if ocr_engine is None:
        ocr_engine = MangaTextOCR()

    return ocr_engine

def get_translator_engine() -> TextTranslator:
    """
    Carga el traductor una sola vez.
    """
    global translator_engine

    if translator_engine is None:
        translator_engine = TextTranslator(
            source_lang="ja",
            target_lang="es"
        )

    return translator_engine

def get_bubble_editor() -> BubbleEditor:
    """
    Carga el editor de globos una sola vez.
    """
    global bubble_editor

    if bubble_editor is None:
        bubble_editor = BubbleEditor()

    return bubble_editor

with gr.Blocks(title="Manga Translator MVP") as demo:
    gr.Markdown(
        """
        # Manga Translator MVP
        
        Sistema de traducción asistida de manga usando visión por computadora, OCR y traducción automática.
        """
    )

    image_input = gr.Image(
        label="Sube una imagen de manga",
        type="pil"
    )

    with gr.Row():
        load_button = gr.Button("Cargar imagen")
        detect_button = gr.Button("Procesar imagen")

    info_output = gr.Textbox(
        label="Información de la imagen",
        lines=4
    )

    with gr.Tabs():
        with gr.Tab("1. Imagen y detección"):
            with gr.Row():
                image_loaded = gr.Image(
                    label="Imagen cargada",
                    type="pil"
                )

                detected_image = gr.Image(
                    label="Imagen con globos detectados",
                    type="pil"
                )

            detection_output = gr.Textbox(
                label="Resultados de detección",
                lines=12
            )

        with gr.Tab("2. Globos recortados"):
            cropped_gallery = gr.Gallery(
                label="Globos recortados",
                columns=2,
                rows=2,
                height=500,
                object_fit="contain"
            )

        with gr.Tab("3. OCR y traducción"):
            with gr.Row():
                ocr_output = gr.Textbox(
                    label="Texto extraído por OCR",
                    lines=18
                )

                translation_output = gr.Textbox(
                    label="Traducción automática al español",
                    lines=18
                )

        with gr.Tab("4. Limpieza visual"):
            clean_gallery = gr.Gallery(
                label="Globos con texto original limpiado",
                columns=2,
                rows=2,
                height=500,
                object_fit="contain"
            )
            
        with gr.Tab("5. Globos traducidos"):
            translated_gallery = gr.Gallery(
                label="Globos con traducción insertada",
                columns=2,
                rows=2,
                height=500,
                object_fit="contain"
            )
            
        with gr.Tab("6. Imagen final traducida"):
            final_translated_image = gr.Image(
                label="Página final con globos traducidos",
                type="pil"
            )
            
    status_output = gr.Textbox(
        label="Estado del procesamiento",
        lines=10,
        interactive=False
    )

    load_button.click(
        fn=load_manga_image,
        inputs=image_input,
        outputs=[image_loaded, info_output]
    )
    
    clear_button = gr.ClearButton(
        components=[
            image_input,
            image_loaded,
            detected_image,
            detection_output,
            cropped_gallery,
            ocr_output,
            translation_output,
            clean_gallery,
            translated_gallery,
            final_translated_image,
            status_output
        ],
        value="Limpiar resultados"
    )

    detect_button.click(
    fn=detect_bubbles,
    inputs=image_input,
    outputs=[
        detected_image,
        detection_output,
        cropped_gallery,
        ocr_output,
        translation_output,
        clean_gallery,
        translated_gallery,
        final_translated_image,
        status_output
    ]

)


if __name__ == "__main__":
    demo.launch()