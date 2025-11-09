#!/usr/bin/env python3
"""
Send a single video file (webm/mp4) to the translate-asl chunk API.

Usage:
  python send_translate_chunk.py --file /path/to/chunk.webm \
    --url http://127.0.0.1:8000/v1/translate_asl/chunk --token YOUR_KEY

The script posts multipart/form-data with field 'video' (filename preserved),
and optional form fields 'session_id' and 'chunk_index'. It prints the JSON
response or the HTTP error body.
"""

import argparse
import os
import sys
import time
import requests
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def send_file(path: str, url: str, token: str | None = None, session_id: str | None = None, chunk_index: int | None = None, max_retries: int = 3):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(1, max_retries + 1):
        try:
            with open(path, "rb") as f:
                files = {"video": (os.path.basename(path), f, "video/webm")}
                data = {}
                if session_id is not None:
                    data["session_id"] = session_id
                if chunk_index is not None:
                    data["chunk_index"] = str(chunk_index)

                logger.info(f"Posting {path} to {url} (attempt {attempt})")
                r = requests.post(url, files=files, data=data, headers=headers, timeout=60)

            if r.ok:
                try:
                    print(r.json())
                except Exception:
                    print(r.text)
                return r
            else:
                logger.warning(f"Server returned {r.status_code}: {r.text}")
                # retry on 5xx
                if 500 <= r.status_code < 600 and attempt < max_retries:
                    time.sleep(1 * attempt)
                    continue
                return r
        except requests.RequestException as e:
            logger.warning(f"Request failed: {e}")
            if attempt < max_retries:
                time.sleep(1 * attempt)
                continue
            raise


def parse_args():
    p = argparse.ArgumentParser(description="Send a video chunk to the translate_asl chunk endpoint")
    p.add_argument("--file", required=True, help="Path to video file to upload")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/translate_asl/chunk", help="Endpoint URL")
    p.add_argument("--token", default=os.getenv("GEMINI_API_TOKEN") or os.getenv("GENAI_API_KEY"), help="Optional bearer token or set GEMINI_API_TOKEN/GENAI_API_KEY env")
    p.add_argument("--session-id", default=None, help="Optional session id to include in form data")
    p.add_argument("--chunk-index", type=int, default=None, help="Optional chunk index")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        send_file(args.file, args.url, token=args.token, session_id=args.session_id, chunk_index=args.chunk_index)
    except Exception as e:
        logger.exception("Failed to send chunk")
        sys.exit(1)
