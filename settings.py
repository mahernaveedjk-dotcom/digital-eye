import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_NAME = "Camera 1"
CAMERA_INDEX = 0

DB_PATH = os.path.join(BASE_DIR, "digital_eye.db")
ZONES_FILE = os.path.join(BASE_DIR, "zones.json")

INCIDENTS_DIR = os.path.join(BASE_DIR, "incidents")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
CONTINUOUS_DIR = os.path.join(RECORDINGS_DIR, "continuous")
CLIPS_DIR = os.path.join(RECORDINGS_DIR, "clips")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Zone settings
GRACE = 1.5
USE_FEET = False
WARNING_SENDS_EMAIL = False

# Detection settings
PERSON_EVERY = 2
DISPLAY_FPS = 15
DETECTION_SIZES = {
    "Fast": 320,
    "Balanced": 416,
    "Accurate": 640
}

# Camera settings
RECONNECT_WAIT = 2.0
WEAPON_GONE_SECONDS = 5

# Recording settings
RECORD_FPS = 20
MAX_RECORD_SECONDS = 1800
CHUNK_SECONDS = 300
CLIP_BEFORE = 10
CLIP_AFTER = 20
MAX_CLIP_SECONDS = 180
MIN_FREE_GB = 2

# Weapon model
WEAPON_REPO = "Subh775/Threat-Detection-YOLOv8n"
WEAPON_FILE = "weights/best.pt"
LOCAL_WEAPON_MODEL = os.path.join(MODELS_DIR, "weapon_best.pt")
WEAPON_CLASSES = {"gun", "knife", "grenade"}

# Drawing settings (OpenCV uses BGR colors)
RESTRICTED_COLOR = (0, 0, 255)
WARNING_COLOR = (0, 200, 255)
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX

# Create required folders
for folder in (
    INCIDENTS_DIR,
    RECORDINGS_DIR,
    CONTINUOUS_DIR,
    CLIPS_DIR,
    MODELS_DIR
):
    os.makedirs(folder, exist_ok=True)