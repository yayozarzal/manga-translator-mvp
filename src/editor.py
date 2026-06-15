from typing import List, Dict
import re
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class BubbleEditor:
    """
    Editor visual para limpiar texto dentro de globos de manga
    e insertar la traducción dentro del globo limpio.
    """

    def __init__(self):
        pass

    def clean_text_from_bubble(self, bubble_image: Image.Image) -> Image.Image:
        """
        Limpia texto oscuro dentro del globo evitando borrar bordes.

        Estrategia:S
        - Detecta píxeles oscuros.
        - Analiza componentes conectados.
        - Borra componentes pequeños/interiores que parecen texto.
        - Conserva componentes grandes o pegados a los bordes, que suelen ser el contorno del globo.
        """
        image_rgb = bubble_image.convert("RGB")
        image_np = np.array(image_rgb)

        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Detectar zonas oscuras: texto, bordes y trazos.
        _, dark_mask = cv2.threshold(
            gray,
            190,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Suaviza un poco ruido, pero sin engrosar demasiado los bordes.
        kernel = np.ones((2, 2), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark_mask,
            connectivity=8
        )

        text_mask = np.zeros((h, w), dtype=np.uint8)

        # Banda de protección: todo lo pegado al borde del recorte se conserva.
        border_margin = max(int(min(w, h) * 0.06), 6)

        for label in range(1, num_labels):
            x, y, bw, bh, area = stats[label]
            cx, cy = centroids[label]

            if area <= 2:
                continue

            touches_border = (
                x <= border_margin or
                y <= border_margin or
                x + bw >= w - border_margin or
                y + bh >= h - border_margin
            )

            # Los bordes de globos suelen ser largos, grandes o tocar el borde del recorte.
            too_large = area > (w * h * 0.08)
            too_wide = bw > (w * 0.65)
            too_tall = bh > (h * 0.65)

            # Evitar borrar contornos o líneas grandes.
            if touches_border or too_large or too_wide or too_tall:
                continue

            # Zona preferente de texto: parte central del globo.
            inside_center = (
                border_margin < cx < w - border_margin and
                border_margin < cy < h - border_margin
            )

            if inside_center:
                text_mask[labels == label] = 255

        # Dilatar un poco para cubrir completamente las letras.
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        cleaned_np = image_np.copy()
        cleaned_np[text_mask > 0] = [255, 255, 255]

        return Image.fromarray(cleaned_np)

    def clean_bubbles(self, translated_results: List[Dict]) -> List[Dict]:
        """
        Aplica limpieza visual a cada globo recortado.
        """
        cleaned_results = []

        for result in translated_results:
            crop = result["crop"]

            try:
                clean_crop = self.clean_text_from_bubble(crop)
            except Exception:
                clean_crop = crop

            cleaned_results.append({
                **result,
                "clean_crop": clean_crop
            })

        return cleaned_results

    def sanitize_translated_text(self, text: str) -> str:
        """
        Limpia el texto traducido antes de insertarlo en el globo.
        """
        if not text:
            return ""

        text = text.strip()
        text = unicodedata.normalize("NFKC", text)

        replacements = {
            "…": "...",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "「": "",
            "」": "",
            "『": "",
            "』": "",
            "【": "",
            "】": "",
            "〜": "",
            "・": "",
            "□": "",
            "■": "",
            "▯": "",
            "\n": " ",
            "\t": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Elimina caracteres japoneses residuales
        text = re.sub(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", "", text)

        # Permite caracteres normales del español
        allowed_pattern = r"[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9.,;:!?¡¿()\"' -]"
        text = re.sub(allowed_pattern, "", text)

        # Limpia espacios repetidos
        text = re.sub(r"\s+", " ", text).strip()

        # Corrige espacios antes de signos
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)

        # Corrige espacios después de signos de apertura
        text = re.sub(r"([¡¿])\s+", r"\1", text)

        # Elimina puntos, comas, guiones, dos puntos, etc. al inicio
        text = re.sub(r"^[\.\,\-\:\;_]+", "", text).strip()

        # Elimina secuencias de puntos al inicio: "...", "..", "."
        text = re.sub(r"^\.+", "", text).strip()

        # Elimina combinaciones raras de puntuación al inicio antes de una palabra
        text = re.sub(r"^[!¡?¿\-\.,;:]+(?=[A-Za-zÁÉÍÓÚáéíóúÑñÜü])", "", text).strip()

        return text

    def get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """
        Carga una fuente disponible en Windows.
        Se priorizan fuentes legibles y estéticas para manga.
        """
        font_candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",   # Segoe UI Bold
            "C:/Windows/Fonts/arialbd.ttf",    # Arial Bold
            "C:/Windows/Fonts/calibrib.ttf",   # Calibri Bold
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]

        for font_path in font_candidates:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

        return ImageFont.load_default()

    def wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        """
        Divide el texto en varias líneas para que no se salga del globo.
        """
        words = text.split()

        if not words:
            return [""]

        lines = []
        current_line = ""

        dummy_img = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy_img)

        for word in words:
            test_line = word if not current_line else f"{current_line} {word}"
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def text_block_size(
        self,
        lines: List[str],
        font: ImageFont.ImageFont,
        line_spacing: int = 5
    ) -> tuple[int, int]:
        """
        Calcula el tamaño total que ocupará el bloque de texto.
        """
        dummy_img = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy_img)

        max_width = 0
        total_height = 0

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]

            max_width = max(max_width, line_width)
            total_height += line_height + line_spacing

        if lines:
            total_height -= line_spacing

        return max_width, total_height

    def draw_centered_text(
        self,
        image: Image.Image,
        text: str,
        min_font_size: int = 11,
        max_font_size: int = 24
    ) -> Image.Image:
        """
        Inserta texto centrado dentro del globo limpio.
        Ajusta automáticamente tamaño de fuente y saltos de línea.
        """
        image_rgb = image.convert("RGB")
        draw = ImageDraw.Draw(image_rgb)

        w, h = image_rgb.size

        # Área segura para escribir dentro del globo.
        margin_x = max(int(w * 0.20), 14)
        margin_y = max(int(h * 0.16), 14)

        max_text_width = max(w - (2 * margin_x), 20)
        max_text_height = max(h - (2 * margin_y), 20)

        clean_text = self.sanitize_translated_text(text)

        if not clean_text:
            return image_rgb

        selected_font = self.get_font(min_font_size)
        selected_lines = [clean_text]
        selected_line_spacing = 5

        # Buscar el mayor tamaño de fuente que quepa en el globo.
        for font_size in range(max_font_size, min_font_size - 1, -1):
            font = self.get_font(font_size)

            # Mientras más grande la fuente, más espaciado suave.
            line_spacing = max(int(font_size * 0.18), 4)

            lines = self.wrap_text(clean_text, font, max_text_width)
            block_width, block_height = self.text_block_size(
                lines,
                font,
                line_spacing=line_spacing
            )

            if block_width <= max_text_width and block_height <= max_text_height:
                selected_font = font
                selected_lines = lines
                selected_line_spacing = line_spacing
                break

        block_width, block_height = self.text_block_size(
            selected_lines,
            selected_font,
            line_spacing=selected_line_spacing
        )

        start_y = (h - block_height) // 2
        current_y = start_y

        for line in selected_lines:
            bbox = draw.textbbox((0, 0), line, font=selected_font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]

            x = (w - line_width) // 2

            draw.text(
                (x, current_y),
                line,
                font=selected_font,
                fill=(0, 0, 0)
            )

            current_y += line_height + selected_line_spacing

        return image_rgb

    def insert_translations(self, cleaned_results: List[Dict]) -> List[Dict]:
        """
        Inserta la traducción en cada globo limpio.
        Agrega una nueva clave: translated_crop.
        """
        final_results = []

        for result in cleaned_results:
            clean_crop = result["clean_crop"]
            translated_text = result.get("translated_text", "")

            try:
                translated_crop = self.draw_centered_text(
                    image=clean_crop,
                    text=translated_text,
                    min_font_size=11,
                    max_font_size=24
                )
            except Exception:
                translated_crop = clean_crop

            final_results.append({
                **result,
                "translated_crop": translated_crop
            })

        return final_results

    def compose_translated_page(
        self,
        original_image: Image.Image,
        final_results: List[Dict]
    ) -> Image.Image:
        """
        Reconstruye la página original pegando cada globo traducido
        en su posición correspondiente.
        """
        page = original_image.convert("RGB").copy()

        for result in final_results:
            translated_crop = result.get("translated_crop")
            coordinates = result.get("coordinates", {})

            if translated_crop is None or not coordinates:
                continue

            x1 = coordinates["x1"]
            y1 = coordinates["y1"]
            x2 = coordinates["x2"]
            y2 = coordinates["y2"]

            target_width = x2 - x1
            target_height = y2 - y1

            if target_width <= 0 or target_height <= 0:
                continue

            translated_crop = translated_crop.convert("RGB")

            translated_crop = translated_crop.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )

            page.paste(translated_crop, (x1, y1))

        return page
    
    def build_center_ellipse_mask(self, w: int, h: int) -> np.ndarray:
        """
        Máscara elíptica de respaldo cuando no se detecta bien el globo.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        center = (w // 2, h // 2)
        axes = (
            max(int(w * 0.30), 10),
            max(int(h * 0.38), 10)
        )

        cv2.ellipse(
            mask,
            center=center,
            axes=axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1
        )

        return mask


    def extract_bubble_mask(self, image_np: np.ndarray) -> np.ndarray:
        """
        Intenta extraer una máscara del interior del globo a partir de las zonas claras.
        Si falla, usa una máscara elíptica de respaldo.
        """
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Detectar zonas claras (interior blanco del globo)
        _, bright_mask = cv2.threshold(
            gray,
            220,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

        center_x, center_y = w // 2, h // 2
        selected_label = -1
        best_area = 0

        for label in range(1, num_labels):
            x, y, bw, bh, area = stats[label]

            if area < 200:
                continue

            # Preferir componentes que contengan el centro
            if x <= center_x <= x + bw and y <= center_y <= y + bh:
                if area > best_area:
                    best_area = area
                    selected_label = label

        if selected_label == -1:
            return self.build_center_ellipse_mask(w, h)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[labels == selected_label] = 255

        # Suavizar bordes
        kernel2 = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

        return mask