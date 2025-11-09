from fastapi import FastAPI
from backend.routes import router
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

app = FastAPI()
app.include_router(router=router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
