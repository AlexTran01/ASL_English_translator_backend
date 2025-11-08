from fastapi import APIRouter
from .translate_asl import router as translate_router
from .simple_ui import router as ui_router
# from .gemini import router as gemini_router

router = APIRouter()

router.include_router(translate_router)
router.include_router(ui_router)
# router.include_router(gemini_router)

