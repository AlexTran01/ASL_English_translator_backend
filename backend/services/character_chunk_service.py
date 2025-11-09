import os
import tempfile
import logging
from typing import List

import numpy as np
import cv2
import joblib
import mediapipe as mp

from backend.const.constant import CHARACTER_PREDICTOR_PKL, CHARACTER_SCALER_PKL, HAND_LANDMARKER_ASSET

logger = logging.getLogger(__name__)

# Log configured asset locations for easier debugging
try:
    logger.info("Character model dir config:")
    logger.info(f"  PREDICTOR_PKL = {CHARACTER_PREDICTOR_PKL}")
    logger.info(f"  SCALER_PKL    = {CHARACTER_SCALER_PKL}")
    logger.info(f"  HAND_ASSET    = {HAND_LANDMARKER_ASSET}")
    logger.info(f"  predictor exists: {CHARACTER_PREDICTOR_PKL.exists()}")
    logger.info(f"  scaler exists:    {CHARACTER_SCALER_PKL.exists()}")
    logger.info(f"  hand asset exists: {HAND_LANDMARKER_ASSET.exists()}")
except Exception:
    logger.exception("Failed while logging model asset paths")

# Load classifier & scaler
classifier = None
scaler = None
try:
    if os.path.exists(CHARACTER_PREDICTOR_PKL) and os.path.exists(CHARACTER_SCALER_PKL):
        classifier = joblib.load(CHARACTER_PREDICTOR_PKL)
        scaler = joblib.load(CHARACTER_SCALER_PKL)
    else:
        logger.warning("Character predictor/scaler pickles not found at configured paths")
except Exception:
    logger.exception("Failed to load classifier or scaler pickles")

# Prepare MediaPipe HandLandmarker (IMAGE mode for synchronous predict)
HandLandmarker = mp.tasks.vision.HandLandmarker
_landmarker = None
try:
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_ASSET)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.4,
    )
    _landmarker = HandLandmarker.create_from_options(options)
except Exception:
    logger.exception("Failed to initialize HandLandmarker; ensure mediapipe and the task file exist")


def _decode_image_bytes(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _extract_first_frame_from_video_bytes(data: bytes, suffix: str = ".webm") -> np.ndarray | None:
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="asl_demo_")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(tmp)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        return None
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def _classify_landmarks(landmarks_list) -> List[str]:
    preds: List[str] = []
    if classifier is None or scaler is None:
        raise RuntimeError("Classifier or scaler not initialized")

    for idx, lm in enumerate(landmarks_list):
        flat = []
        try:
            items = None
            if hasattr(lm, 'landmark'):
                items = lm.landmark
            else:
                try:
                    items = list(lm)
                except Exception:
                    items = None

            if items is None:
                raise ValueError('No landmark items to iterate')

            for l in items:
                x = getattr(l, 'x', None)
                y = getattr(l, 'y', None)
                z = getattr(l, 'z', None)
                if x is not None and y is not None and z is not None:
                    flat.extend([float(x), float(y), float(z)])
                    continue

                # otherwise, try to treat l as a sequence
                try:
                    seq = list(l)
                    if len(seq) >= 3:
                        flat.extend([float(seq[0]), float(seq[1]), float(seq[2])])
                        continue
                except Exception:
                    pass

                # last resort: if l itself is numeric
                try:
                    flat.append(float(l))
                except Exception:
                    # unknown element type; include a placeholder and continue
                    logger.debug(f"Unexpected landmark element type at idx {idx}: {type(l)}")
                    flat.append(0.0)

        except Exception:
            # fallback: attempt numpy flatten (may still result in non-floats)
            try:
                flat = [float(x) for x in np.array(lm).flatten()]
            except Exception:
                flat = []

        handedness = 0
        # calling code should supply handedness if desired; we set 0 as default
        features = [handedness] + flat

        try:
            # defensive checks before passing to scaler
            if len(features) == 0 or all(isinstance(x, (list, tuple)) for x in features):
                raise ValueError('Extracted features are empty or malformed')

            X = scaler.transform(np.array([features]))
            p = classifier.predict(X)
            preds.append(str(p[0]))
        except Exception:
            # Log feature diagnostics to help debugging
            try:
                sample_types = [type(x).__name__ for x in features[:10]]
                logger.exception(f"Failed to run classifier on extracted landmarks; features_len={len(features)}, sample_types={sample_types}")
            except Exception:
                logger.exception("Failed to run classifier on extracted landmarks (and failed to collect diagnostics)")
            preds.append("ERROR")
    return preds


def predict_from_upload_bytes(data: bytes, filename: str | None = None, content_type: str | None = None) -> List[str]:
    """
    Accept raw upload bytes (image or video). Return list of predicted labels (one per detected hand).

    Returns:
        list of prediction strings (one per detected hand)
    """
    if _landmarker is None:
        raise RuntimeError("HandLandmarker not initialized")

    img = None
    if content_type and content_type.startswith("image/"):
        img = _decode_image_bytes(data)
    elif content_type and content_type.startswith("video/"):
        suffix = os.path.splitext(filename or "")[1] or ".webm"
        img = _extract_first_frame_from_video_bytes(data, suffix=suffix)
    else:
        # try image first
        img = _decode_image_bytes(data)
        if img is None:
            suffix = os.path.splitext(filename or "")[1] or ".webm"
            img = _extract_first_frame_from_video_bytes(data, suffix=suffix)

    if img is None:
        raise RuntimeError("Could not decode upload into an image frame")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    if h != w:
        size = max(h, w)
        # create square canvas and center the original image
        square = np.zeros((size, size, 3), dtype=rgb.dtype)
        top = (size - h) // 2
        left = (size - w) // 2
        square[top:top + h, left:left + w] = rgb
        rgb = square

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = _landmarker.detect(mp_img)
    landmarks_ls = getattr(result, 'hand_world_landmarks', []) or []
    handedness_ls = getattr(result, 'handedness', []) or []

    preds: List[str] = []
    for i, lm in enumerate(landmarks_ls):
        handedness = 0
        try:
            handedness = handedness_ls[i][0].index
        except Exception:
            handedness = 0
        try:
            classification = _classify_landmarks([lm])
            preds.append(classification[0])
        except Exception:
            preds.append("ERROR")

    return preds
