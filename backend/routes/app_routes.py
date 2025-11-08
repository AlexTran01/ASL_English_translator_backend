from fastapi import APIRouter, UploadFile, File, HTTPException, status
from backend.controllers.app_controller import translate_asl
import tempfile
import os


router = APIRouter()


@router.post("/translate_asl", response_model=str)
async def translate_route(video: UploadFile = File(...)):
	"""Accept an uploaded video file, save it temporarily, call the controller, and return the predicted label."""
	# Basic validation
	if not video.filename:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

	suffix = os.path.splitext(video.filename)[1] or ".mp4"
	try:
		contents = await video.read()
		with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="asl_upload_") as tmp:
			tmp.write(contents)
			tmp_path = tmp.name

		# Log the saved file path and size to help debugging when frames are empty
		try:
			size = os.path.getsize(tmp_path)
		except Exception:
			size = -1
		print(f"Saved uploaded video to: {tmp_path} (size={size} bytes)")

		result = translate_asl(tmp_path)

		return result

	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to process video: {e}")

	finally:
		try:
			if 'tmp_path' in locals() and os.path.exists(tmp_path):
				os.remove(tmp_path)
		except Exception:
			pass


@router.get("/", response_model=dict)
async def root():
	return {"message": "ASL to English Translator API is running."}