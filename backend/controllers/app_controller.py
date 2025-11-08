import torch
import torchvision
import torch.nn as nn
from backend.utils.video_converter import video_to_tensor

# Load checkpoint (attempt CPU by default)
CHECKPOINT = 'backend/ai_modules/Model_words_level/final_hackathons_model.pth'

# Choose device: prefer CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(CHECKPOINT, map_location=device)

# Build/load model robustly from checkpoint
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    model = checkpoint['model']
elif isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
    # recreate model architecture if needed
    # fallback to a ResNet3D skeleton — adjust if your training used a different arch
    model = torchvision.models.video.r3d_18(weights=None)
    num_classes = len(checkpoint.get('class_mapping', [])) or model.fc.out_features
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
    model.load_state_dict(state)
else:
    if hasattr(checkpoint, 'parameters'):
        model = checkpoint
    else:
        raise RuntimeError(f'Unrecognized checkpoint format: {type(checkpoint)}')

if hasattr(model, 'module'):
    model = model.module

# Move model to the chosen device
model = model.to(device)
model.eval()

# class mapping lookup
class_mapping = None
if isinstance(checkpoint, dict):
    class_mapping = checkpoint.get('class_mapping')

def translate_asl(vidpath):
    print("translate_asl is called")
    
    clips = video_to_tensor(video_path=vidpath)  # expected shape (1, C, T, H, W)
    print("clips shape:", clips.shape)
    print("clips sample", clips[0,:,0, :, :])  # print first frame of the first clip
    # Ensure batch dim
    if clips.dim() == 4:
        clips = clips.unsqueeze(0)
    
    # Move inputs to model device
    clips = clips.to(device)

    with torch.no_grad():
        logits = model(clips)
        pred_idx = logits.argmax(1).item()

    # Map to label if mapping exists
    if class_mapping:
        try:
            pred_label = class_mapping[pred_idx]
        except Exception:
            if isinstance(class_mapping, dict):
                inv = {v: k for k, v in class_mapping.items()}
                pred_label = inv.get(pred_idx, str(pred_idx))
            else:
                pred_label = str(pred_idx)
    else:
        pred_label = str(pred_idx)

    print(f'Predicted class: {pred_label}')
    return pred_label
