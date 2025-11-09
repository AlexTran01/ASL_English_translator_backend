import os
import cv2
import torch
import numpy as np
import tempfile
import subprocess

from backend.const.constant import MAX_FRAMES
from backend.utils.data_preprocessing import VideoTransform


def video_to_tensor(video_path: str, num_frames: int = MAX_FRAMES, to_rgb: bool = True) -> torch.Tensor:
    """
    Read a video or image from disk, sample frames, apply project VideoTransform and return a tensor.

    Returns:
        torch.Tensor: shape (1, C, T, H, W), dtype float32.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video_to_tensor: file not found: {video_path}")

    def _is_image(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")

    def _try_convert_with_ffmpeg(src_path: str) -> str | None:
        fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="cv_convert_")
        os.close(fd)
        cmd = [
            "ffmpeg", "-y",
            "-i", src_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            out_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            return out_path
        except Exception:
            try:
                os.remove(out_path)
            except Exception:
                pass
            return None

    frames: list[np.ndarray] = []
    cleanup_converted: str | None = None

    # Single image → single-frame "video"
    if _is_image(video_path):
        img = cv2.imread(video_path)
        if img is None:
            raise RuntimeError(f"video_to_tensor: failed to read image {video_path}")
        frames = [img]
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            converted = _try_convert_with_ffmpeg(video_path)
            if converted:
                cap = cv2.VideoCapture(converted)
                cleanup_converted = converted
            else:
                raise RuntimeError(f"video_to_tensor: cannot open video file with OpenCV: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        else:
            frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frame_idx_set = set(frame_idxs.tolist())
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in frame_idx_set:
                    frames.append(frame)

        try:
            cap.release()
        except Exception:
            pass

        if cleanup_converted:
            try:
                os.remove(cleanup_converted)
            except Exception:
                pass
    print(f"video_to_tensor: collected {len(frames)} frames from {video_path}")
    if frames:
        try:
            first = frames[0]
            print(
                f"first frame shape: {getattr(first, 'shape', None)} "
                f"dtype: {getattr(first, 'dtype', None)} "
                f"min/max: {np.min(first)} / {np.max(first)}"
            )
        except Exception as e:
            print(f"video_to_tensor: failed to inspect first frame: {e}")

    transform = VideoTransform(max_frames=num_frames, train=False)

    if not frames:
        print(f"video_to_tensor: warning — no frames captured from {video_path}; returning zeros")
        # safer: bypass transform when no frames
        # adjust (C, T, H, W) to your actual expected shape
        C, T, H, W = 3, num_frames, 112, 112
        t = torch.zeros((C, T, H, W), dtype=torch.float32)
        return t.unsqueeze(0)

    arr = np.array(frames)  # (T, H, W, C), BGR
    if to_rgb:
        arr = arr[..., ::-1]  # BGR -> RGB

    if arr.dtype != np.uint8:
        try:
            arr = arr.astype(np.uint8)
        except Exception:
            print(f"video_to_tensor: warning — cannot cast frames to uint8, dtype={arr.dtype}")

    t = transform(arr)  # expect (C, T, H, W)

    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)

    t = t.unsqueeze(0)  # (1, C, T, H, W)
    return t
