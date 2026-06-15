from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# Modelos
MODEL_DIR = BASE_DIR / "models"

OWN_MODEL_PATH = MODEL_DIR / "bubble_detector.pt"
EXTERNAL_MODEL_PATH = MODEL_DIR / "comic-speech-bubble-detector.pt"

MODEL_OPTIONS = {
    "Modelo propio YOLOv8": str(OWN_MODEL_PATH),
    "Modelo externo Hugging Face": str(EXTERNAL_MODEL_PATH),
}

DEFAULT_MODEL_NAME = "Modelo propio YOLOv8"

# Detección
CONFIDENCE_THRESHOLD = 0.30
DETECTION_CLASS_ID = 0
MIN_BOX_WIDTH = 15
MIN_BOX_HEIGHT = 15

# Recorte
CROP_PADDING = 0
CROP_SHRINK_RATIO = 0.08

# Procesamiento de imagen
MAX_IMAGE_SIDE = 3000
MIN_IMAGE_SIDE = 100
MAX_IMAGE_PIXELS = 20_000_000

# Edición de texto
MIN_FONT_SIZE = 11
MAX_FONT_SIZE = 24

# Reproducibilidad
RANDOM_SEED = 42

# Directorios
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
DEMO_DIR = BASE_DIR / "data" / "demo"

for directory in [INPUT_DIR, OUTPUT_DIR, DEMO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)