from backend.services.translation_service import default_service

model = default_service.model
device = default_service.device
class_mapping = default_service.class_mapping


def translate_asl(video_path: str) -> str:
    """Translate a video by delegating to the TranslationService.

    Keep the function signature identical to previous code so routes and
    scripts remain unchanged.
    """
    return default_service.predict(video_path)
