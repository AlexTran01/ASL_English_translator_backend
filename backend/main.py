from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return{"message": "backend Live"}

@app.post("/translate")
def translate(data: Item ):
    
    return {"translate" : data.value}


