from fastapi import UploadFile
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

def video_to_tensor(video_path, num_frames=8, resize=(224, 224)):
    """
    Convert video to tensor shape (C, T, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)  # pick evenly spaced frames
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_idxs:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, resize)
            frames.append(frame)
    
    cap.release()
    
    # Convert to tensor
    frames = np.array(frames)  # shape (T, H, W, C)
    frames = frames.transpose((3, 0, 1, 2))  # (C, T, H, W)
    frames = torch.tensor(frames, dtype=torch.float32) / 255.0  # normalize 0-1
    
    # Add batch dimension
    frames = frames.unsqueeze(0)  # (1, C, T, H, W)
    
    return frames
