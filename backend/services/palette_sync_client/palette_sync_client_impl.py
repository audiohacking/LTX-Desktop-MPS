"""HTTP implementation of PaletteSyncClient."""

from __future__ import annotations

from typing import Any, cast

from services.http_client.http_client import HTTPClient


class PaletteSyncClientImpl:
    def __init__(self, http: HTTPClient, base_url: str = "https://directorspalette.com") -> None:
        self._http = http
        self._base_url = base_url

    def validate_connection(self, *, api_key: str) -> dict[str, Any]:
        resp = self._http.get(
            f"{self._base_url}/api/desktop/auth/validate",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Palette auth failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())

    def get_credits(self, *, api_key: str) -> dict[str, Any]:
        resp = self._http.get(
            f"{self._base_url}/api/desktop/credits",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Palette credits failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())
