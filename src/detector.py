from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path


class BubbleDetector:
    """
    Detector de globos/cuadros de texto en imágenes de manga usando YOLO.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.30
    ):
        model_file = Path(model_path)

        if not model_file.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo de detección en: {model_file}"
            )

        self.model_path = str(model_file)
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(self.model_path)
    

    def detect(self, image: Image.Image) -> Tuple[Image.Image, List[Dict]]:
        """
        Detecta globos/cuadros de texto en una imagen.

        Args:
            image: Imagen en formato PIL.

        Returns:
            annotated_image: Imagen con cajas dibujadas.
            detections: Lista de detecciones con coordenadas y confianza.
        """

        image_rgb = image.convert("RGB")
        image_np = np.array(image_rgb)

        results = self.model.predict(
            source=image_np,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []
        annotated_np = image_np.copy()

        if not results:
            return image_rgb, detections

        result = results[0]

        if result.boxes is None:
            return image_rgb, detections

        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            class_name = self.model.names.get(class_id, f"class_{class_id}")

            detection = {
                "id": index + 1,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            }

            detections.append(detection)

            cv2.rectangle(
                annotated_np,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{detection['class_name']} {confidence:.2f}"

            cv2.putText(
                annotated_np,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            detections = sorted(
                detections,
                key=lambda det: (det["y1"], det["x1"])
        )

            for index, det in enumerate(detections):
                det["id"] = index + 1

        annotated_image = Image.fromarray(annotated_np)

        return annotated_image, detections
    
    def crop_detections(self, image: Image.Image, detections: List[Dict], padding: int = 0, shrink_ratio: float = 0.08) -> List[Dict]:
        """
        Recorta las regiones detectadas en la imagen original.

        padding:
            Margen adicional. Para modelos que detectan justo, usar 8-12.
            Para modelos que detectan de más, usar 0.

        shrink_ratio:
            Reduce internamente el bounding box.
            Ej: 0.08 reduce 8% por cada lado aprox.
        """
        image_rgb = image.convert("RGB")
        width, height = image_rgb.size

        cropped_bubbles = []

        for det in detections:
            x1_original = det["x1"]
            y1_original = det["y1"]
            x2_original = det["x2"]
            y2_original = det["y2"]

            box_w = x2_original - x1_original
            box_h = y2_original - y1_original

            shrink_x = int(box_w * shrink_ratio)
            shrink_y = int(box_h * shrink_ratio)

            x1 = max(x1_original + shrink_x - padding, 0)
            y1 = max(y1_original + shrink_y - padding, 0)
            x2 = min(x2_original - shrink_x + padding, width)
            y2 = min(y2_original - shrink_y + padding, height)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_rgb.crop((x1, y1, x2, y2))

            cropped_bubbles.append({
                "id": det["id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "original_coordinates": {
                    "x1": x1_original,
                    "y1": y1_original,
                    "x2": x2_original,
                    "y2": y2_original,
                },
                "crop": crop
            })

        return cropped_bubbles
            