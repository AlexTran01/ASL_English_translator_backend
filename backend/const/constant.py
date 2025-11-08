from pathlib import Path
import os

# Project root (backend/)
ROOT = Path(__file__).resolve().parents[1]

# Static and upload directories (inside backend/)
STATIC_DIR = ROOT / 'static'
UPLOAD_DIR = ROOT / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoint default path
CHECKPOINT_PATH = ROOT / 'ai_modules' / 'Model_words_level' / 'Word_level_model_1.pth' / 'Word_level_model_2.pth'

# Video processing
MAX_FRAMES = 16

KEEP_UPLOADS = True
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB default

# Misc
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000
