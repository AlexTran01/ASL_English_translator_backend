#!/usr/bin/env python3
"""
Usage:
  python record_and_send_chunks.py --duration 2 --url http://127.0.0.1:8000/gemini_asl/chunk --token "$GEMINI_API_KEY"

Options:
  --duration    Chunk length in seconds (default: 2)
  --camera      Camera index (default: 0)
  --save        If set, keep recorded chunk files locally. Otherwise they are deleted after sending.
  --token       Optional bearer token for Authorization header.
  --url         Server URL to POST chunks to. Default: http://127.0.0.1:8000/translate_asl/chunk

This script uses OpenCV to capture and write MP4 files using the 'mp4v' codec.
"""

import argparse
import cv2
import tempfile
import os
import time
import requests
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def send_chunk(path: str, url: str, token: str | None = None, max_retries: int = 3) -> requests.Response | None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(1, max_retries + 1):
        try:
            with open(path, "rb") as f:
                files = {"video": (os.path.basename(path), f, "video/mp4")}
                logger.info(f"Posting chunk {os.path.basename(path)} to {url} (attempt {attempt})")
                r = requests.post(url, files=files, headers=headers, timeout=30)
            logger.info(f"Response status: {r.status_code}")
            try:
                logger.info(f"Response text: {r.text}")
            except Exception:
                pass
            return r
        except Exception as e:
            logger.warning(f"Failed to send chunk (attempt {attempt}): {e}")
            time.sleep(1 * attempt)
    logger.error(f"Giving up sending {path} after {max_retries} attempts")
    return None


def record_and_send(url: str = "http://127.0.0.1:8000/gemini_asl/chunk", chunk_duration: float = 2.0, camera_index: int = 0, save_chunks: bool = False, token: str | None = None):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera. Exiting.")
        sys.exit(1)

    # Read camera properties; some drivers report an incorrect/very low FPS (e.g. 1.0).
    # Treat very-low reported FPS as unreliable and use a sane default.
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps < 5.0:
        logger.info(f"Camera reported low/unreliable FPS: {fps}. Using default 30.0 FPS for writer.")
        writer_fps = 30.0
    else:
        writer_fps = fps
        logger.info(f"Camera reported FPS: {fps}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    logger.info(f"Capture resolution: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    chunk_index = 0

    try:
        while True:
            chunk_index += 1
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            tmpf = tempfile.NamedTemporaryFile(prefix=f"chunk_{ts}_", suffix=".mp4", delete=False)
            tmpf.close()
            tmp_path = tmpf.name

            logger.info(f"Recording chunk #{chunk_index} -> {tmp_path}")
            writer = cv2.VideoWriter(tmp_path, fourcc, writer_fps, (width, height))
            frames_written = 0

            start = time.time()
            while True:
                now = time.time()
                elapsed = now - start
                if elapsed >= chunk_duration:
                    break

                ret, frame = cap.read()
                if not ret:
                    logger.debug("Frame read returned False; retrying shortly")
                    time.sleep(0.05)
                    continue

                writer.write(frame)
                frames_written += 1

            writer.release()
            elapsed = time.time() - start
            logger.info(f"Finished recording chunk #{chunk_index}: {frames_written} frames, {elapsed:.2f}s")

            # Send the chunk to server
            resp = send_chunk(tmp_path, url, token)

            if not save_chunks:
                try:
                    os.remove(tmp_path)
                    logger.debug(f"Deleted temp chunk {tmp_path}")
                except Exception:
                    logger.warning(f"Could not delete temp file {tmp_path}")

            if elapsed < chunk_duration:
                to_sleep = max(0, chunk_duration - elapsed)
                logger.debug(f"Sleeping {to_sleep:.3f}s to preserve chunk cadence")
                time.sleep(to_sleep)

    except KeyboardInterrupt:
        logger.info("Interrupted by user — stopping capture")
    finally:
        cap.release()
        logger.info("Camera released. Exiting.")


def parse_args():
    p = argparse.ArgumentParser(description="Continuously record short video chunks and POST to a server endpoint.")
    p.add_argument("--duration", type=float, default=2.0, help="Chunk length in seconds (default 2)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--url", type=str, default="http://127.0.0.1:8000/translate_asl/chunk", help="Endpoint to POST chunks to")
    p.add_argument("--save", action="store_true", help="Keep chunks on disk instead of deleting them")
    p.add_argument("--token", type=str, default=os.getenv("GEMINI_API_KEY", None), help="Optional bearer token for Authorization header (or set GEMINI_API_KEY env)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Starting recorder -> {args.url} (chunk duration {args.duration}s)")
    record_and_send(url=args.url, chunk_duration=args.duration, camera_index=args.camera, save_chunks=args.save, token=args.token)
