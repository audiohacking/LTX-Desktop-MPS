"""Model downloader service protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class ModelDownloader(Protocol):
    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
        expected_size_bytes: int | None = None,
    ) -> Path: ...

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
        expected_size_bytes: int | None = None,
    ) -> Path: ...
