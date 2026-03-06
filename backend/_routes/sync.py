"""Palette sync routes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/sync", tags=["sync"])


@router.get("/status")
def sync_status(handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    return handler.sync.get_status()


@router.get("/credits")
def sync_credits(handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    return handler.sync.get_credits()
