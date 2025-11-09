from pathlib import Path
import os

# Project root (backend/)
ROOT = Path(__file__).resolve().parents[1]

# Static and upload directories (inside backend/)
STATIC_DIR = ROOT / 'static'
UPLOAD_DIR = ROOT / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoint default path
CHECKPOINT_PATH = ROOT / 'ai_modules' / 'Model_words_level' / 'Word_level_model_2.pth'

# Video processing
MAX_FRAMES = 16

KEEP_UPLOADS = True
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB default

# Misc
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000

# Character-level model assets (used by demo / hand-landmarker)
CHARACTER_MODEL_DIR = ROOT / 'ai_modules' / 'Model_character_level'
HAND_LANDMARKER_ASSET = CHARACTER_MODEL_DIR / 'hand_landmarker.task'
CHARACTER_PREDICTOR_PKL = CHARACTER_MODEL_DIR / 'predictor.pkl'
CHARACTER_SCALER_PKL = CHARACTER_MODEL_DIR / 'scaler.pkl'

# Prompt
SYSTEM_PROMPT = """You are an expert ASL interpreter.
        Analyze the uploaded video and identify the **word-level ASL gesture** being performed **BY HAND**.

        Describe step-by-step:
        1. The **handshape(s)** (open hand, closed fist, flat palm, etc.)
        2. The **hand location** relative to the body (e.g., near head, chest, etc.)
        3. The **movement direction or repetition** (e.g., forward, circular, tapping, etc.)
        4. The **facial expression or posture** if visible
        5. Finally, give the **most likely English equivalent word/ phrase** for the sign.

        If uncertain, give the top 2 possible interpretations, ordered by confidence.

        Answer concisely, e.g.:
        Gesture: [WORD] e.g., "HELLO", "THANK YOU", "LOVE",...
        Description: [brief motion and shape summary]
        Confidence: [percentage or range]
        """