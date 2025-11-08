from fastapi import UploadFile
import cv2
import torch
import numpy as np
import os
import tempfile
import subprocess
from backend.const.const import MAX_FRAMES
from backend.utils.data_preprocessing import VideoTransform


def video_to_tensor(video_path, num_frames=MAX_FRAMES, resize=None):
    """Read a video from disk, sample frames, apply project VideoTransform and return a tensor

    Returns tensor with shape (1, C, T, H, W) and dtype float32.
    """
    # Validate path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'video_to_tensor: file not found: {video_path}')

    def _is_image(path):
        return os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp')

    def _try_convert_with_ffmpeg(src_path):
        fd, out_path = tempfile.mkstemp(suffix='.mp4', prefix='cv_convert_')
        os.close(fd)
        cmd = ['ffmpeg', '-y', '-i', src_path, '-c:v', 'libx264', '-preset', 'veryfast', '-movflags', '+faststart', out_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_path
        except Exception:
            try:
                os.remove(out_path)
            except Exception:
                pass
            return None

    frames = []
    cleanup_converted = None

    # If input is a single image, read and return it as single-frame
    if _is_image(video_path):
        img = cv2.imread(video_path)
        if img is None:
            raise RuntimeError(f'video_to_tensor: failed to read image {video_path}')
        frames = [img]
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Try ffmpeg conversion (useful for webm/mkv containers)
            converted = _try_convert_with_ffmpeg(video_path)
            if converted:
                cap = cv2.VideoCapture(converted)
                cleanup_converted = converted
            else:
                raise RuntimeError(f'video_to_tensor: cannot open video file with OpenCV: {video_path}')

        # Try to sample frames by index if available, otherwise read sequentially
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # unknown length, read all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        else:
            frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in frame_idxs:
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

    # If no frames captured, return zeros shaped to (1, C, T, H, W)
    transform = VideoTransform(max_frames=num_frames, train=False)

    if len(frames) == 0:
        t = transform(np.zeros((0, 112, 112, 3), dtype=np.uint8))
    else:
        arr = np.array(frames)  # (T, H, W, C) in BGR
        t = transform(arr)  # returns (C, T, H, W)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)

    t = t.unsqueeze(0)  # (1, C, T, H, W)
    return t
