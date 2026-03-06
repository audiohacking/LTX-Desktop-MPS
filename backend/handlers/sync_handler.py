"""Handler for Palette sync operations."""
from __future__ import annotations

import logging
from typing import Any

from services.palette_sync_client.palette_sync_client import PaletteSyncClient
from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class SyncHandler:
    def __init__(self, state: AppState, palette_sync_client: PaletteSyncClient) -> None:
        self._state = state
        self._client = palette_sync_client
        self._cached_user: dict[str, Any] | None = None

    def get_status(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "user": None}
        try:
            user = self._client.validate_connection(api_key=api_key)
            self._cached_user = user
            return {"connected": True, "user": user}
        except Exception:
            self._cached_user = None
            return {"connected": False, "user": None}

    def get_credits(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "balance": None}
        try:
            credits = self._client.get_credits(api_key=api_key)
            return {"connected": True, **credits}
        except Exception:
            return {"connected": False, "balance": None}
