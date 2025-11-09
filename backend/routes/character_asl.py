import logging
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.services.character_chunk_service import predict_from_upload_bytes

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/v1/character_internal_model_asl/chunk')
async def hand_predict(request: Request):
    """Accept multipart/form-data with the file field named either `file` or `video`.

    This keeps compatibility with the frontend which may append chunks as `video`.
    """
    form = await request.form()
    upload = None
    # Prefer 'file' then 'video'
    if 'file' in form:
        upload = form['file']
    elif 'video' in form:
        upload = form['video']

    if upload is None:
        raise HTTPException(status_code=400, detail="No file field in form; expected 'file' or 'video'")

    # upload can be an UploadFile-like object
    try:
        filename = getattr(upload, 'filename', None)
        content_type = getattr(upload, 'content_type', None)
        data = await upload.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    try:
        preds = predict_from_upload_bytes(data, filename=filename, content_type=content_type)
        return JSONResponse({"predictions": preds, "num_hands": len(preds)})
    except RuntimeError as e:
        logger.exception("Model/service not initialized")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

