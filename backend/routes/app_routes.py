from fastapi import APIRouter
from controllers.app_controller import translate_asl


router = APIRouter()

router.post("/translate_asl") (translate_asl)