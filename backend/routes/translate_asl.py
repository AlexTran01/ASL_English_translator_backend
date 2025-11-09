import os
import time
from backend.const.constant import UPLOAD_DIR, KEEP_UPLOADS
from backend.utils.file_io import save_upload_file, rename_to_label, remove_file
from backend.models.schemas import PredictionResponse, StatusResponse
from backend.controllers.app_controller import translate_asl as translate_asl_controller, translate_asl_bytes
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request, Form


router = APIRouter()

@router.get("/", response_model=StatusResponse)
async def root():
	return StatusResponse(message="ASL to English Translator API is running.")

@router.post("/translate_asl/chunk", response_model=PredictionResponse)
async def translate_asl(video: UploadFile = File(...)):
	"""Accept an uploaded video file,
	  save it temporarily, 
	  call the controller, 
	  and return the predicted label."""
	if not video.filename:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

	suffix = os.path.splitext(video.filename)[1] or ".mp4"

	try:
		# If KEEP_UPLOADS is enabled we preserve the previous behavior and
		# stream the upload to disk. Otherwise, read bytes directly from the
		# UploadFile and run inference without saving a persistent file.
		if KEEP_UPLOADS:
			tmp_path = await save_upload_file(video, UPLOAD_DIR)
			print(f"Saved uploaded video to: {tmp_path}")

			# call the controller function (aliased to avoid name shadowing with this route)
			result = translate_asl_controller(tmp_path)

			try:
				tmp_path = rename_to_label(tmp_path, result)
				print(f"Renamed uploaded file to prediction name: {tmp_path}")
			except Exception as e:
				print(f"Failed to rename uploaded file with prediction: {e}")

		else:
			# read bytes and pass directly to model (no permanent save)
			video_bytes = await video.read()
			result = translate_asl_bytes(video_bytes)

		return PredictionResponse(label=str(result))

	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to process video: {e}")

	finally:
		if not KEEP_UPLOADS:
			try:
				if 'tmp_path' in locals() and os.path.exists(tmp_path):
					remove_file(tmp_path)
			except Exception:
				pass

@router.post("/v1/translate_asl/chunk", response_model=PredictionResponse)
async def translate_asl_v1(request: Request, video: UploadFile = File(...), session_id: str | None = Form(default=None), chunk_index: int | None = Form(default=None)):
	start_time = time.time()
	
	if not (video.content_type and video.content_type.startswith("video/")):
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")

	video_bytes = await video.read()
	result = translate_asl_bytes(video_bytes)

	return PredictionResponse(label=str(result), processing_time_ms=int((time.time() - start_time) * 1000))