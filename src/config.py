from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Input source
# 0 = webcam
# "video.mp4" = local video file
VIDEO_SOURCE = "demo.mp4"

# Demo / debug
ENABLE_DEBUG_MOCK_FLAME = True

# Configuration constants

# Timeouts in seconds
TIMEOUT_SECONDS = 300
WARNING_SECONDS = 240
CRITICAL_SECONDS = 300

# Detection settings
PERSON_CONFIDENCE_THRESHOLD = 0.40
FLAME_CONFIDENCE_THRESHOLD = 0.35
FIRE_CONFIDENCE_THRESHOLD = 0.35

# Safe Zones Settings
BURNER_RADIUS_PIXELS = 100
BURNERS_FILE = "burner_zones.json"

# Temporal / Growth Settings
GROWTH_WARNING_MULTIPLIER = 1.5
GROWTH_CRITICAL_MULTIPLIER = 2.0
BASELINE_FRAMES = 300
CURRENT_AREA_FRAMES = 30
BASELINE_DECAY_RATE = 0.005

# Model paths
PERSON_MODEL_PATH = "yolov8n.pt"
FLAME_MODEL_PATH = str(MODELS_DIR / "stove_fire_best.pt")
FIRE_MODEL_PATH = str(MODELS_DIR / "fire_best.pt")

SHUTOFF_ALPHA = 0.04      # Controls EMA smoothing speed
SHUTOFF_THRESHOLD = 0.85   # Confidence required to hard-trigger shutdown