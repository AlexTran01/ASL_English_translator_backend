import os
import cv2
import torch
import numpy as np
import tempfile

from backend.const.constant import MAX_FRAMES
from backend.utils.data_preprocessing import VideoTransform


def video_to_tensor(
        video_bytes: bytes | None = None,
        num_frames: int = MAX_FRAMES,
        to_rgb: bool = True,
        tmp_suffix: str = ".webm") -> torch.Tensor:
    """
    Returns:
        torch.Tensor: shape (1, C, T, H, W), dtype float32.
    """
    if not video_bytes:
        raise ValueError("Either video_path or video_bytes must be provided")
    
    fd, tmp_path = tempfile.mkstemp(suffix=tmp_suffix, prefix="asl_chunk_")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError(f"video_bytes_to_tensor: cannot open temp video {tmp_path}")

        frames: list[np.ndarray] = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        else:
            idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            idx_set = set(idxs.tolist())
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in idx_set:
                    frames.append(frame)

        cap.release()

        if not frames:
            C, T, H, W = 3, num_frames, 112, 112
            return torch.zeros((1, C, T, H, W), dtype=torch.float32)

        arr = np.stack(frames, axis=0)

        if to_rgb:
            arr = arr[..., ::-1]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        transform = VideoTransform(max_frames=num_frames, train=False)
        t = transform(arr)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)

        if t.ndim == 4:
            t = t.unsqueeze(0)
        return t.float()

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
   