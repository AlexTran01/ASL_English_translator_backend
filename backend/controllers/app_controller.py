import torch
import torchvision
import torch.nn as nn
from utils.video_converter import video_to_tensor

checkpoint = torch.load('backend/ai_modules/Model_words_level/final_hackathons_model.pth', map_location='cpu')
model = torchvision.models.video.r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(checkpoint['class_mapping']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
class_mapping = checkpoint['class_mapping']

def translate_asl(vidpath):
    print("translate_asl is called")
    
    # Example prediction
    clips = video_to_tensor(video_path=vidpath) # preprocessed tensor (C,T,H,W)
    with torch.no_grad():
        logits = model(clips)
        pred_idx = logits.argmax(1).item()
        pred_label = class_mapping[pred_idx]

    print(f'Predicted class: {pred_label}')
    return pred_label
