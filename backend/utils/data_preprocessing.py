import numpy as np
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from backend.const.const import MAX_FRAMES

# in backend/utils/data_preprocessing.py
class VideoTransform:
    def __init__(self, max_frames=MAX_FRAMES, train=False):
        self.max_frames = max_frames
        if train:
            self.aug = A.Compose([
                A.RandomResizedCrop((112,112), scale=(0.8,1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
                A.ColorJitter(p=0.5),
                A.Normalize(mean=(0.45,0.45,0.45), std=(0.225,0.225,0.225)),
                ToTensorV2()
            ])
        else:
            # deterministic eval transform
            self.aug = A.Compose([
                A.Resize(112,112),
                A.CenterCrop(112,112),
                A.Normalize(mean=(0.45,0.45,0.45), std=(0.225,0.225,0.225)),
                ToTensorV2()
            ])

    def __call__(self, clip):
        frames = []
        for i in range(min(self.max_frames, clip.shape[0])):
            frm = clip[i]
            try:
                rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                t = self.aug(image=rgb)['image']  # (C,H,W)
                if t.ndim == 3:
                    frames.append(t)
            except:
                continue
        if not frames:
            return torch.zeros((3, self.max_frames, 112, 112), dtype=torch.float)
        if len(frames) < self.max_frames:
            frames = (frames * ((self.max_frames // len(frames)) + 1))[:self.max_frames]
        else:
            frames = frames[:self.max_frames]
        stacked = torch.stack(frames)  # (T,C,H,W)
        return stacked.permute(1,0,2,3)  # (C,T,H,W)