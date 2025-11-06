from fastapi import APIRouter
from controllers.app_controller import translate_asl


app = APIRouter()

app.post("/translate_asl") (translate_asl)