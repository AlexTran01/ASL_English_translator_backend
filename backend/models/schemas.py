from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    """API response for a single prediction."""
    label: str
    confidence: Optional[float] = None

class StatusResponse(BaseModel):
    message: str

class ChunkPrediction(BaseModel):
    session_id: str | None = None
    chunk_index: int | None = None
    label: str
    confidence: float | None = None
    alternatives: list[PredictionResponse] | None = None
    raw_text: str | None = None

class ChunkResponse(BaseModel):
    prediction: ChunkPrediction
    model: str = "models/gemini-2.5-flash"
    processing_time_ms: int | None = None

