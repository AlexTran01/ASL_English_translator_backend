import torch

from typing import Optional, Any
from backend.const.constant import CHECKPOINT_PATH
from backend.utils.video_converter import video_to_tensor
from backend.utils.model_utils import get_model_checkpoint, load_model_state_dict, class_mapping


class TranslationService:
    """
    This service loads a pre-trained ASL translation model and provides
    a method to predict the label of a given video file.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[torch.nn.Module] = None
        self.class_mapping: Optional[Any] = None
        self._load_model()

    def _load_model(self):
        checkpoint_path = get_model_checkpoint(self.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = load_model_state_dict(checkpoint)

        # Unwrap DataParallel if necessary
        if hasattr(model, 'module'):
            model = model.module

        model = model.to(self.device)
        model.eval()

        self.model = model
        if isinstance(checkpoint, dict):
            self.class_mapping = checkpoint.get('class_mapping')

    def predict(self, video_path: str) -> str:
        """Run inference on a single video file and return a predicted label string."""
        if self.model is None:
            raise RuntimeError('Model is not loaded')

        clips = video_to_tensor(video_path)  # (1, C, T, H, W)
        if clips.dim() == 4:
            clips = clips.unsqueeze(0)

        clips = clips.to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(clips)
            pred_idx = int(logits.argmax(1).item())

        pred_idx = class_mapping(self.class_mapping, pred_idx)
        return pred_idx


default_service = TranslationService()
