import json

from google import genai
from google.genai import types
from backend.const.constant import SYSTEM_PROMPT
from backend.models.schemas import ChunkResponse, ChunkPrediction

def process_gemini_asl_chunk(client: genai.Client, video_bytes: bytes, session_id: str = None, chunk_index: int = None) -> ChunkResponse:
    response = client.models.generate_content(
    model='models/gemini-2.5-flash',
    contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text=SYSTEM_PROMPT)
            ]
        )
    )
    return response

def process_response(response: ChunkResponse, session_id: str | None = None, chunk_index: int | None = None) -> str:
    raw_text = getattr(response, 'text', "") or ""

    label = "UNKNOWN"
    confidence = None
    alternatives = None

    try:
        parsed = json.loads(raw_text)
        label = parsed.get("label", "UNKNOWN")
        confidence = parsed.get("confidence")
        alts = parsed.get("alternatives") or []
        alternatives = [
            AlternativePrediction(
                label=alt.get("label", "UNKNOWN"),
                confidence=alt.get("confidence"),
            )
            for alt in alts
        ]
    except Exception:
        # fallback: just use raw text as label
        label = raw_text.strip()[:64] or "UNKNOWN"

    chunk_prediction = ChunkPrediction(
        session_id=session_id,
        chunk_index=chunk_index,
        label=label,
        confidence=confidence,
        alternatives=alternatives,
        raw_text=raw_text,
    )
    return chunk_prediction