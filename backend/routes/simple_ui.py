from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from backend.const.constant import STATIC_DIR

router = APIRouter()

# Mount static directory if present (mounting here keeps route grouping local to UI)
if Path(STATIC_DIR).is_dir():
    router.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@router.get('/ui', response_class=FileResponse)
async def ui_index():
    """Serve the static index.html (webcam recorder UI)."""
    index_path = Path(STATIC_DIR) / 'index.html'
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "UI not found"}, status_code=404)


@router.get('/', response_class=FileResponse)
async def root_index():
    return JSONResponse({"message": "ASL to English Translator API is running."})
