from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.routes.app_routes import router
import uvicorn
import os

app = FastAPI()

# Mount static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if os.path.isdir(STATIC_DIR):
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

@app.get('/ui')
async def ui_index():
    """Serve the static index.html (webcam recorder UI)."""
    index_path = os.path.join(STATIC_DIR, 'index.html')
    return FileResponse(index_path)

app.include_router(router=router)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
