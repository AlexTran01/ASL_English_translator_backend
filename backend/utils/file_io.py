from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import UploadFile


async def save_upload_file(upload: UploadFile, dest_dir: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream an UploadFile to a new temporary file inside dest_dir.

    Returns the absolute path to the saved file as a string.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = os.path.splitext(upload.filename)[1] or ".mp4"
    tmp_name = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = dest_dir / tmp_name

    async with aiofiles.open(str(tmp_path), 'wb') as out_f:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            await out_f.write(chunk)

    return str(tmp_path)


def sanitize_label(label: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(label))


def rename_to_label(path: str, label: str) -> str:
    """Rename the file at `path` to `<label>.<ext>` inside the same directory.

    If the target exists, append a numeric suffix: label_1.ext, label_2.ext, ...
    Returns the new absolute path as a string.
    """
    p = Path(path)
    safe_label = sanitize_label(label)
    new_name = f"{safe_label}{p.suffix}"
    new_path = p.with_name(new_name)

    idx = 1
    base = safe_label
    while new_path.exists():
        new_name = f"{base}_{idx}{p.suffix}"
        new_path = p.with_name(new_name)
        idx += 1

    p.rename(new_path)
    return str(new_path)


def remove_file(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass
