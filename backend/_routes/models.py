"""Route handlers for /api/models, /api/models/status, /api/models/download/*."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api_types import (
    CheckPathRequest,
    CheckPathResponse,
    DownloadProgressResponse,
    ModelDownloadRequest,
    ModelDownloadStartResponse,
    ModelInfo,
    ModelsStatusResponse,
    TextEncoderDownloadResponse,
)
from _routes._errors import HTTPError
from state import get_state_service
from app_handler import AppHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=list[ModelInfo])
def route_models_list(handler: AppHandler = Depends(get_state_service)) -> list[ModelInfo]:
    return handler.models.get_models_list()


@router.get("/models/status", response_model=ModelsStatusResponse)
def route_models_status(handler: AppHandler = Depends(get_state_service)) -> ModelsStatusResponse:
    return handler.models.get_models_status()


@router.post("/models/check-path", response_model=CheckPathResponse)
def route_models_check_path(
    req: CheckPathRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CheckPathResponse:
    """Scan for required models under the given path. Used by first-run 'I already have models' flow."""
    return handler.models.check_models_at_path(req.comfyuiModelsPath)


@router.get("/models/download/progress", response_model=DownloadProgressResponse)
def route_download_progress(handler: AppHandler = Depends(get_state_service)) -> DownloadProgressResponse:
    return handler.downloads.get_download_progress()


@router.post("/models/download", response_model=ModelDownloadStartResponse)
def route_model_download(
    req: ModelDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ModelDownloadStartResponse:
    if handler.downloads.is_download_running():
        raise HTTPError(409, "Download already in progress")

    settings = handler.settings.get_settings_snapshot()
    skip_text_encoder = req.skipTextEncoder
    if settings.ltx_api_key and not settings.use_local_text_encoder:
        skip_text_encoder = True

    if handler.downloads.start_model_download(skip_text_encoder=skip_text_encoder):
        return ModelDownloadStartResponse(
            status="started",
            message="Model download started",
            skippingTextEncoder=skip_text_encoder,
        )

    raise HTTPError(400, "Failed to start download")


@router.post("/text-encoder/download", response_model=TextEncoderDownloadResponse)
def route_text_encoder_download(handler: AppHandler = Depends(get_state_service)) -> TextEncoderDownloadResponse:
    if handler.downloads.is_download_running():
        raise HTTPError(409, "Download already in progress")

    files = handler.models.refresh_available_files()
    if files["text_encoder"] is not None:
        return TextEncoderDownloadResponse(status="already_downloaded", message="Text encoder already downloaded")

    if handler.downloads.start_text_encoder_download():
        return TextEncoderDownloadResponse(status="started", message="Text encoder download started")

    raise HTTPError(400, "Failed to start download")
