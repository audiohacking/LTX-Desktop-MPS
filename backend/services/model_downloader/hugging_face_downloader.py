"""Hugging Face model download service wrapper."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Lock
from typing import Any
from unittest.mock import patch

from huggingface_hub import file_download, hf_hub_download, snapshot_download  # type: ignore[reportUnknownVariableType]
from tqdm.auto import tqdm as tqdm_auto  # type: ignore[reportUnknownVariableType]


def _make_progress_tqdm_class(callback: Callable[[int, int], None]) -> type:
    """Return a tqdm subclass that reports aggregated progress via *callback*.

    ``snapshot_download`` spawns one tqdm instance per file in the snapshot.
    All instances created from a single factory call share this mutable dict
    so the callback reports total progress across all files in that snapshot.
    """
    # Shared across all tqdm instances created within one download call.
    # Protected by a lock because snapshot_download runs parallel threads.
    lock = Lock()
    shared: dict[str, int] = {"downloaded": 0, "total": 0}

    class _ProgressTqdm(tqdm_auto):  # type: ignore[reportUntypedBaseClass]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)  # type: ignore[reportUnknownMemberType]
            if self.total is not None:
                with lock:
                    shared["total"] += int(self.total)

        def update(self, n: float | int | None = 1) -> bool | None:  # type: ignore[reportIncompatibleMethodOverride]
            result = super().update(n)
            if n is not None:
                with lock:
                    shared["downloaded"] += int(n)
                    downloaded, total = shared["downloaded"], shared["total"]
                callback(downloaded, total)
            return result

    return _ProgressTqdm


@contextlib.contextmanager
def _patch_http_get_progress(
    callback: Callable[[int, int], None],
    expected_size_bytes: int | None = None,
) -> Iterator[None]:
    """Temporarily monkey-patch ``huggingface_hub.file_download.http_get``
    to inject a custom tqdm bar that forwards progress to *callback*.

    ``hf_hub_download`` does not expose a ``tqdm_class`` parameter (unlike
    ``snapshot_download``), but its internal ``http_get`` accepts a private
    ``_tqdm_bar`` kwarg.  We wrap ``http_get`` to inject our own bar when
    the caller hasn't already provided one. If *expected_size_bytes* is set,
    the tqdm is created with that total so progress reports a known total from
    the start (avoids progress bar appearing stuck when the server delays
    sending Content-Length).

    See ``test_http_get_accepts_tqdm_bar`` — if that test breaks after a
    huggingface_hub upgrade, this patch needs to be revisited.
    """
    tqdm_cls = _make_progress_tqdm_class(callback)
    original_http_get: Callable[..., Any] = file_download.http_get  # type: ignore[reportUnknownMemberType]
    tqdm_kw: dict[str, Any] = {"disable": True}
    if expected_size_bytes is not None and expected_size_bytes > 0:
        tqdm_kw["total"] = expected_size_bytes

    def _wrapped_http_get(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("_tqdm_bar") is None:
            kwargs["_tqdm_bar"] = tqdm_cls(**tqdm_kw)
        return original_http_get(*args, **kwargs)

    with patch.object(file_download, "http_get", _wrapped_http_get):
        yield


class HuggingFaceDownloader:
    """Wraps huggingface_hub download functions."""

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
        expected_size_bytes: int | None = None,
    ) -> Path:
        ctx = (
            _patch_http_get_progress(on_progress, expected_size_bytes)
            if on_progress is not None
            else contextlib.nullcontext()
        )
        with ctx:
            path: str = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return Path(path)

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
        expected_size_bytes: int | None = None,
    ) -> Path:
        ctx = (
            _patch_http_get_progress(on_progress, expected_size_bytes)
            if on_progress is not None
            else contextlib.nullcontext()
        )
        with ctx:
            path: str = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        return Path(path)
