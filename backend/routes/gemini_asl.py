import os
import time
from google import genai
from backend.services.gemini_chunk_service import process_gemini_asl_chunk, process_response
from backend.models.schemas import ChunkResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form, Request
from starlette.concurrency import run_in_threadpool
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def _process_chunk_sync(api_key: str | None, video_bytes: bytes, session_id: str | None, chunk_index: int | None):
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client()

    try:
        return process_gemini_asl_chunk(client, video_bytes, session_id, chunk_index)
    finally:
        try:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            logger.debug("genai client close failed", exc_info=True)


@router.post("/v1/gemini_asl/chunk", response_model=ChunkResponse)
async def gemini_asl(request: Request, video: UploadFile = File(...), session_id: str | None = Form(default=None), chunk_index: int | None = Form(default=None)):
    start_time = time.time()
    if not (video.content_type and video.content_type.startswith("video/")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")

    try:
        auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
        api_key = None
        if auth_header and isinstance(auth_header, str) and auth_header.lower().startswith("bearer "):
            api_key = auth_header.split(" ", 1)[1].strip()

        video_bytes = await video.read()

        response = await run_in_threadpool(_process_chunk_sync, api_key, video_bytes, session_id, chunk_index)

    except Exception as e:
        logger.exception("Failed to process gemini chunk")
        raise HTTPException(status_code=500, detail=f"Failed to process video chunk: {e}")

    chunk_prediction = process_response(response, session_id, chunk_index)

    return ChunkResponse(prediction=chunk_prediction, processing_time_ms=int((time.time() - start_time) * 1000))
