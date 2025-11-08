import os
import torch
import torchvision
import torch.nn as nn

def get_model_checkpoint(checkpoint_path: str) -> str:
    """Get the model checkpoint path from environment variable or use default."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path

def load_model_state_dict(checkpoint):
    """Load the model from the checkpoint."""
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    elif isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
        model = torchvision.models.video.r3d_18(weights=None)
        num_classes = len(checkpoint.get('class_mapping', [])) or model.fc.out_features
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        state = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
        model.load_state_dict(state)
        return model
    else:
        if hasattr(checkpoint, 'parameters'):
            return checkpoint
        else:
            raise RuntimeError(f'Unrecognized checkpoint format: {type(checkpoint)}')
        

def class_mapping(class_mapping, pred_idx):
    if class_mapping:
        try:
            if isinstance(class_mapping, dict):
                vals = list(class_mapping.values())
                if all(isinstance(v, int) for v in vals):
                    inv = {v: k for k, v in class_mapping.items()}
                    return inv.get(pred_idx, str(pred_idx))
                else:
                    return class_mapping.get(pred_idx, str(pred_idx))
            elif isinstance(class_mapping, (list, tuple)):
                return class_mapping[pred_idx] if 0 <= pred_idx < len(class_mapping) else str(pred_idx)
        except Exception:
            return str(pred_idx)
    return str(pred_idx)
