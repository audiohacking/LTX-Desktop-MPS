"""Serialize LTX GPU pipeline forwards across threads.

``ltx2_server`` starts a background warmup thread and a queue worker. Both can
invoke ``DistilledPipeline`` (or siblings) concurrently on the same process.
That is unsafe: ``LTXTextEncoder``'s ModelLedger patch temporarily sets
``model_ledger.device`` to CPU while building the cached text encoder; a second
thread can observe that state, or restore ``device`` to CPU in ``finally`` when
its ``saved_device`` was already CPU — leaving transformer weights on CPU while
latents stay on MPS (``RuntimeError: weight is on cpu but expected on mps``).
PyTorch modules are not thread-safe for parallel ``forward`` either.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import TypeVar

_F = TypeVar("_F", bound=Callable[..., object])

ltx_pipeline_forward_lock = threading.Lock()


def with_ltx_pipeline_forward_lock(fn: _F) -> _F:
    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        with ltx_pipeline_forward_lock:
            return fn(*args, **kwargs)

    return wrapped  # type: ignore[return-value]
