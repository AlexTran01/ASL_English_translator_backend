from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    """API response for a single prediction."""
    label: str
    confidence: Optional[float] = None


class StatusResponse(BaseModel):
    message: str
