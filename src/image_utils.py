from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 3000
MIN_IMAGE_SIDE = 100


def validate_image(image: Optional[Image.Image]) -> None:
    """
    Valida que la entrada sea una imagen procesable.
    """
    if image is None:
        raise ValueError("Debe seleccionar una imagen antes de procesar.")

    if not isinstance(image, Image.Image):
        raise TypeError("El archivo recibido no es una imagen válida.")

    width, height = image.size

    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise ValueError(
            f"La imagen es demasiado pequeña: {width}x{height}px. "
            f"El lado mínimo permitido es {MIN_IMAGE_SIDE}px."
        )

    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            "La imagen supera el límite de 20 millones de píxeles."
        )


def prepare_image(image: Image.Image) -> Tuple[Image.Image, str]:
    """
    Corrige la orientación, convierte a RGB y reduce imágenes demasiado grandes.

    Returns:
        Imagen preparada y mensaje informativo.
    """
    validate_image(image)

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    original_width, original_height = image.size
    message = "Imagen validada correctamente."

    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail(
            (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
            Image.Resampling.LANCZOS
        )

        message = (
            f"Imagen redimensionada de "
            f"{original_width}x{original_height}px a "
            f"{image.width}x{image.height}px."
        )

    return image, message


def normalize_image(image: Image.Image) -> Image.Image:
    """
    Conserva compatibilidad con el código anterior.
    """
    return image.convert("RGB")


def get_image_info(image: Image.Image) -> str:
    width, height = image.size

    return (
        "Imagen preparada correctamente.\n"
        f"Resolución: {width} x {height}px\n"
        f"Modo de color: {image.mode}"
    )


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)