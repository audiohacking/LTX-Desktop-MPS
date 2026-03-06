"""Protocol for communicating with Director's Palette cloud API."""

from __future__ import annotations

from typing import Any, Protocol


class PaletteSyncClient(Protocol):
    def validate_connection(self, *, api_key: str) -> dict[str, Any]:
        """Validate API key and return user info. Raises on failure."""
        ...

    def get_credits(self, *, api_key: str) -> dict[str, Any]:
        """Return credit balance for the authenticated user."""
        ...
