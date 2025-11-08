import os
from backend.const.constant import UPLOAD_DIR, KEEP_UPLOADS
from backend.utils.file_io import save_upload_file, rename_to_label, remove_file
from backend.models.schemas import PredictionResponse, StatusResponse
from backend.controllers.app_controller import translate_asl
from fastapi import APIRouter, UploadFile, File, HTTPException, status


router = APIRouter()

@router.post("/translate_asl", response_model=PredictionResponse)
async def translate_route(video: UploadFile = File(...)):
	"""Accept an uploaded video file,
	  save it temporarily, 
	  call the controller, 
	  and return the predicted label."""
	if not video.filename:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

	suffix = os.path.splitext(video.filename)[1] or ".mp4"

	try:
		tmp_path = await save_upload_file(video, UPLOAD_DIR)
		print(f"Saved uploaded video to: {tmp_path}")

		result = translate_asl(tmp_path)

		if KEEP_UPLOADS:
			try:
				tmp_path = rename_to_label(tmp_path, result)
				print(f"Renamed uploaded file to prediction name: {tmp_path}")
			except Exception as e:
				print(f"Failed to rename uploaded file with prediction: {e}")

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

@router.get("/", response_model=StatusResponse)
async def root():
	return StatusResponse(message="ASL to English Translator API is running.")