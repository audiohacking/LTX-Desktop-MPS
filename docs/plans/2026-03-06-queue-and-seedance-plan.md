# Queue + Seedance 1.5 Pro Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a persistent job queue with dual GPU/API parallel slots, and Seedance 1.5 Pro video model via Replicate.

**Architecture:** Jobs are submitted to a queue (persisted as JSON), processed by a background worker with two concurrent slots (GPU for local, API for Replicate). Seedance 1.5 Pro always routes through the API slot via Replicate. Frontend polls queue status and shows all jobs in the existing results area.

**Tech Stack:** Python FastAPI backend, React TypeScript frontend, Replicate API, Pydantic models

---

## Task 1: Video API Client Protocol + Replicate Implementation

Create the VideoAPIClient service following the same pattern as ImageAPIClient.

**Files:**
- Create: `backend/services/video_api_client/__init__.py`
- Create: `backend/services/video_api_client/video_api_client.py`
- Create: `backend/services/video_api_client/replicate_video_client_impl.py`
- Test: `backend/tests/test_video_api_client.py`

**Step 1: Write the protocol**

`backend/services/video_api_client/video_api_client.py`:
```python
"""Video API client protocol for cloud video generation."""

from __future__ import annotations

from typing import Protocol


class VideoAPIClient(Protocol):
    def generate_text_to_video(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        generate_audio: bool,
    ) -> bytes:
        ...
```

`backend/services/video_api_client/__init__.py`:
```python
from services.video_api_client.video_api_client import VideoAPIClient
from services.video_api_client.replicate_video_client_impl import ReplicateVideoClientImpl

__all__ = ["VideoAPIClient", "ReplicateVideoClientImpl"]
```

**Step 2: Write failing tests for the Replicate video client**

`backend/tests/test_video_api_client.py`:
```python
"""Tests for the Replicate video API client."""

from __future__ import annotations

import pytest

from tests.fakes.services import FakeHTTPClient, FakeResponse
from services.video_api_client.replicate_video_client_impl import ReplicateVideoClientImpl


def _make_client(http: FakeHTTPClient) -> ReplicateVideoClientImpl:
    return ReplicateVideoClientImpl(http=http, api_base_url="https://test.replicate.com/v1")


def test_seedance_text_to_video_sync_success() -> None:
    http = FakeHTTPClient()
    client = _make_client(http)

    # Prediction returns succeeded immediately (Prefer: wait)
    http.queue("post", FakeResponse(
        status_code=201,
        json_payload={
            "id": "pred-1",
            "status": "succeeded",
            "output": "https://example.com/video.mp4",
        },
    ))
    # Download the video
    http.queue("get", FakeResponse(status_code=200, content=b"fake-mp4-bytes"))

    result = client.generate_text_to_video(
        api_key="test-key",
        model="seedance-1.5-pro",
        prompt="a cat dancing",
        duration=8,
        resolution="720p",
        aspect_ratio="16:9",
        generate_audio=True,
    )

    assert result == b"fake-mp4-bytes"
    # Verify the POST was sent to the correct model endpoint
    assert "bytedance/seedance-1.5-pro" in http.calls[0].url
    # Verify input payload
    payload = http.calls[0].json_payload
    assert payload is not None
    assert payload["input"]["prompt"] == "a cat dancing"
    assert payload["input"]["duration"] == 8
    assert payload["input"]["seed"] is not None


def test_seedance_text_to_video_polling_success() -> None:
    http = FakeHTTPClient()
    client = _make_client(http)

    # Prediction returns processing
    http.queue("post", FakeResponse(
        status_code=201,
        json_payload={
            "id": "pred-2",
            "status": "processing",
            "urls": {"get": "https://test.replicate.com/v1/predictions/pred-2"},
        },
    ))
    # First poll: still processing
    http.queue("get", FakeResponse(
        status_code=200,
        json_payload={"id": "pred-2", "status": "processing"},
    ))
    # Second poll: succeeded
    http.queue("get", FakeResponse(
        status_code=200,
        json_payload={
            "id": "pred-2",
            "status": "succeeded",
            "output": "https://example.com/video2.mp4",
        },
    ))
    # Download
    http.queue("get", FakeResponse(status_code=200, content=b"polled-video"))

    result = client.generate_text_to_video(
        api_key="test-key",
        model="seedance-1.5-pro",
        prompt="a dog running",
        duration=4,
        resolution="480p",
        aspect_ratio="9:16",
        generate_audio=False,
    )

    assert result == b"polled-video"


def test_unknown_model_raises() -> None:
    http = FakeHTTPClient()
    client = _make_client(http)

    with pytest.raises(RuntimeError, match="Unknown video model"):
        client.generate_text_to_video(
            api_key="test-key",
            model="nonexistent-model",
            prompt="test",
            duration=4,
            resolution="720p",
            aspect_ratio="16:9",
            generate_audio=False,
        )


def test_prediction_failure_raises() -> None:
    http = FakeHTTPClient()
    client = _make_client(http)

    http.queue("post", FakeResponse(
        status_code=201,
        json_payload={
            "id": "pred-fail",
            "status": "failed",
            "error": "GPU OOM",
        },
    ))

    with pytest.raises(RuntimeError, match="failed"):
        client.generate_text_to_video(
            api_key="test-key",
            model="seedance-1.5-pro",
            prompt="test",
            duration=4,
            resolution="720p",
            aspect_ratio="16:9",
            generate_audio=False,
        )
```

**Step 3: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_video_api_client.py -v --tb=short`
Expected: FAIL (module not found)

**Step 4: Implement ReplicateVideoClientImpl**

`backend/services/video_api_client/replicate_video_client_impl.py`:
```python
"""Replicate API client implementation for cloud video generation (Seedance)."""

from __future__ import annotations

import time
from typing import Any, cast

from services.http_client.http_client import HTTPClient
from services.services_utils import JSONValue

REPLICATE_API_BASE_URL = "https://api.replicate.com/v1"

_MODEL_ROUTES: dict[str, str] = {
    "seedance-1.5-pro": "bytedance/seedance-1.5-pro",
}

_POLL_INTERVAL_SECONDS = 2
_POLL_TIMEOUT_SECONDS = 300


class ReplicateVideoClientImpl:
    def __init__(self, http: HTTPClient, *, api_base_url: str = REPLICATE_API_BASE_URL) -> None:
        self._http = http
        self._base_url = api_base_url.rstrip("/")

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        generate_audio: bool,
    ) -> bytes:
        replicate_model = _MODEL_ROUTES.get(model)
        if replicate_model is None:
            raise RuntimeError(f"Unknown video model: {model}")

        input_payload = self._build_input(
            prompt=prompt,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            generate_audio=generate_audio,
        )

        prediction = self._create_prediction(
            api_key=api_key,
            replicate_model=replicate_model,
            input_payload=input_payload,
        )

        output_url = self._wait_for_output(api_key, prediction)
        return self._download_video(api_key, output_url)

    @staticmethod
    def _build_input(
        *,
        prompt: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        generate_audio: bool,
    ) -> dict[str, JSONValue]:
        seed = int(time.time()) % 2_147_483_647
        return {
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
            "seed": seed,
        }

    def _create_prediction(
        self,
        *,
        api_key: str,
        replicate_model: str,
        input_payload: dict[str, JSONValue],
    ) -> dict[str, Any]:
        url = f"{self._base_url}/models/{replicate_model}/predictions"
        payload: dict[str, JSONValue] = {"input": input_payload}

        response = self._http.post(
            url,
            headers=self._headers(api_key, prefer_wait=True),
            json_payload=payload,
            timeout=300,
        )
        if response.status_code not in (200, 201):
            detail = response.text[:500] if response.text else "Unknown error"
            raise RuntimeError(f"Replicate prediction failed ({response.status_code}): {detail}")

        return self._json_object(response.json(), context="create prediction")

    def _wait_for_output(self, api_key: str, prediction: dict[str, Any]) -> str:
        status = prediction.get("status", "")
        if status == "succeeded":
            return self._extract_output_url(prediction)

        if status in ("failed", "canceled"):
            error = prediction.get("error", "Unknown error")
            raise RuntimeError(f"Replicate prediction {status}: {error}")

        poll_url = prediction.get("urls", {}).get("get")
        if not isinstance(poll_url, str) or not poll_url:
            prediction_id = prediction.get("id", "")
            poll_url = f"{self._base_url}/predictions/{prediction_id}"

        deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_SECONDS)
            resp = self._http.get(poll_url, headers=self._headers(api_key), timeout=30)
            if resp.status_code != 200:
                detail = resp.text[:500] if resp.text else "Unknown error"
                raise RuntimeError(f"Replicate poll failed ({resp.status_code}): {detail}")

            data = self._json_object(resp.json(), context="poll")
            poll_status = data.get("status", "")
            if poll_status == "succeeded":
                return self._extract_output_url(data)
            if poll_status in ("failed", "canceled"):
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Replicate prediction {poll_status}: {error}")

        raise RuntimeError("Replicate video prediction timed out")

    def _download_video(self, api_key: str, url: str) -> bytes:
        download = self._http.get(url, headers=self._headers(api_key), timeout=300)
        if download.status_code != 200:
            detail = download.text[:500] if download.text else "Unknown error"
            raise RuntimeError(f"Replicate video download failed ({download.status_code}): {detail}")
        if not download.content:
            raise RuntimeError("Replicate video download returned empty body")
        return download.content

    @staticmethod
    def _headers(api_key: str, *, prefer_wait: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if prefer_wait:
            headers["Prefer"] = "wait"
        return headers

    @staticmethod
    def _extract_output_url(prediction: dict[str, Any]) -> str:
        output = prediction.get("output")
        if isinstance(output, str) and output:
            return output
        if isinstance(output, list) and output:
            output_list = cast(list[object], output)
            first = output_list[0]
            if isinstance(first, str) and first:
                return first
        raise RuntimeError("Replicate response missing output URL")

    @staticmethod
    def _json_object(payload: object, *, context: str) -> dict[str, Any]:
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        raise RuntimeError(f"Unexpected Replicate {context} response format")
```

**Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_video_api_client.py -v --tb=short`
Expected: All 4 tests PASS

**Step 6: Wire into services/interfaces.py**

Add to `backend/services/interfaces.py`:
- Import: `from services.video_api_client.video_api_client import VideoAPIClient`
- Add `"VideoAPIClient"` to `__all__`

**Step 7: Add FakeVideoAPIClient to test fakes**

Add to `backend/tests/fakes/services.py`:
```python
class FakeVideoAPIClient:
    def __init__(self) -> None:
        self.text_to_video_calls: list[dict[str, Any]] = []
        self.raise_on_text_to_video: Exception | None = None
        self.text_to_video_result = b"fake-seedance-video"

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        generate_audio: bool,
    ) -> bytes:
        self.text_to_video_calls.append({
            "api_key": api_key,
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
        })
        if self.raise_on_text_to_video is not None:
            raise self.raise_on_text_to_video
        return self.text_to_video_result
```

Add field to `FakeServices`:
```python
video_api_client: FakeVideoAPIClient = field(default_factory=FakeVideoAPIClient)
```

**Step 8: Wire VideoAPIClient into AppHandler and ServiceBundle**

Modify `backend/app_handler.py`:
- Add `VideoAPIClient` to imports from `services.interfaces`
- Add `video_api_client: VideoAPIClient` param to `AppHandler.__init__` and store as `self.video_api_client`
- Add `video_api_client: VideoAPIClient` field to `ServiceBundle`
- In `build_default_service_bundle`: import `ReplicateVideoClientImpl`, instantiate `ReplicateVideoClientImpl(http=http)`, add to bundle
- In `build_initial_state`: pass `video_api_client=bundle.video_api_client`

Modify `backend/tests/conftest.py`:
- Add `video_api_client=fake_services.video_api_client` to `ServiceBundle(...)` constructor

**Step 9: Run all tests**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add backend/services/video_api_client/ backend/tests/test_video_api_client.py
git add backend/services/interfaces.py backend/tests/fakes/services.py
git add backend/app_handler.py backend/tests/conftest.py
git commit -m "feat: add VideoAPIClient protocol + Replicate Seedance implementation"
```

---

## Task 2: Settings — Add video_model

**Files:**
- Modify: `backend/state/app_settings.py`
- Modify: `backend/tests/test_settings.py`
- Modify: `settings.json`

**Step 1: Add video_model to AppSettings**

In `backend/state/app_settings.py`:
- Add to `AppSettings`: `video_model: str = "ltx-fast"`
- Add to `SettingsResponse`: `video_model: str = "ltx-fast"`

In `settings.json`: add `"videoModel": "ltx-fast"`

**Step 2: Add test**

In `backend/tests/test_settings.py` add:
```python
def test_video_model_roundtrips(client, test_state):
    resp = client.post("/api/settings", json={"videoModel": "seedance-1.5-pro"})
    assert resp.status_code == 200
    assert test_state.state.app_settings.video_model == "seedance-1.5-pro"

    get_resp = client.get("/api/settings")
    assert get_resp.json()["videoModel"] == "seedance-1.5-pro"
```

**Step 3: Run tests**

Run: `cd backend && uv run pytest tests/test_settings.py -v --tb=short`
Expected: PASS

**Step 4: Commit**

```bash
git add backend/state/app_settings.py backend/tests/test_settings.py settings.json
git commit -m "feat: add video_model setting (ltx-fast | seedance-1.5-pro)"
```

---

## Task 3: Job Queue State — QueueJob + JobQueue

**Files:**
- Create: `backend/state/job_queue.py`
- Test: `backend/tests/test_job_queue.py`

**Step 1: Write the test**

`backend/tests/test_job_queue.py`:
```python
"""Tests for the persistent job queue."""

from __future__ import annotations

import json
from pathlib import Path

from state.job_queue import JobQueue, QueueJob


def test_submit_job_assigns_id_and_status(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(
        job_type="video",
        model="seedance-1.5-pro",
        params={"prompt": "hello"},
        slot="api",
    )
    assert job.id
    assert job.status == "queued"
    assert job.slot == "api"
    assert job.progress == 0


def test_get_all_jobs_returns_ordered(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    j1 = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    j2 = queue.submit(job_type="image", model="z-image-turbo", params={}, slot="gpu")
    jobs = queue.get_all_jobs()
    assert [j.id for j in jobs] == [j1.id, j2.id]


def test_next_queued_for_slot(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    queue.submit(job_type="video", model="seedance-1.5-pro", params={}, slot="api")
    queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")

    gpu_job = queue.next_queued_for_slot("gpu")
    assert gpu_job is not None
    assert gpu_job.slot == "gpu"

    api_job = queue.next_queued_for_slot("api")
    assert api_job is not None
    assert api_job.slot == "api"


def test_update_job_status(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue.update_job(job.id, status="running", progress=50, phase="inference")
    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "running"
    assert updated.progress == 50
    assert updated.phase == "inference"


def test_cancel_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue.cancel_job(job.id)
    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "cancelled"


def test_clear_finished_jobs(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    j1 = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    j2 = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue.update_job(j1.id, status="complete")
    queue.clear_finished()
    remaining = queue.get_all_jobs()
    assert len(remaining) == 1
    assert remaining[0].id == j2.id


def test_persistence_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    queue1 = JobQueue(persistence_path=path)
    job = queue1.submit(job_type="video", model="ltx-fast", params={"prompt": "test"}, slot="gpu")

    queue2 = JobQueue(persistence_path=path)
    loaded = queue2.get_job(job.id)
    assert loaded is not None
    assert loaded.params == {"prompt": "test"}


def test_running_jobs_reset_to_queued_on_load(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    queue1 = JobQueue(persistence_path=path)
    job = queue1.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue1.update_job(job.id, status="running")

    queue2 = JobQueue(persistence_path=path)
    loaded = queue2.get_job(job.id)
    assert loaded is not None
    assert loaded.status == "queued"
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_job_queue.py -v --tb=short`
Expected: FAIL (import error)

**Step 3: Implement JobQueue**

`backend/state/job_queue.py`:
```python
"""Persistent job queue for sequential generation processing."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


@dataclass
class QueueJob:
    id: str
    type: Literal["video", "image"]
    model: str
    params: dict[str, Any]
    status: Literal["queued", "running", "complete", "error", "cancelled"]
    slot: Literal["gpu", "api"]
    progress: int = 0
    phase: str = "queued"
    result_paths: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = ""


class JobQueue:
    def __init__(self, persistence_path: Path) -> None:
        self._path = persistence_path
        self._jobs: list[QueueJob] = []
        self._load()

    def submit(
        self,
        *,
        job_type: str,
        model: str,
        params: dict[str, Any],
        slot: str,
    ) -> QueueJob:
        job = QueueJob(
            id=uuid.uuid4().hex[:8],
            type=job_type,  # type: ignore[arg-type]
            model=model,
            params=params,
            status="queued",
            slot=slot,  # type: ignore[arg-type]
            progress=0,
            phase="queued",
            result_paths=[],
            error=None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._jobs.append(job)
        self._save()
        return job

    def get_all_jobs(self) -> list[QueueJob]:
        return list(self._jobs)

    def get_job(self, job_id: str) -> QueueJob | None:
        for job in self._jobs:
            if job.id == job_id:
                return job
        return None

    def next_queued_for_slot(self, slot: str) -> QueueJob | None:
        for job in self._jobs:
            if job.status == "queued" and job.slot == slot:
                return job
        return None

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        phase: str | None = None,
        result_paths: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        if status is not None:
            job.status = status  # type: ignore[assignment]
        if progress is not None:
            job.progress = progress
        if phase is not None:
            job.phase = phase
        if result_paths is not None:
            job.result_paths = result_paths
        if error is not None:
            job.error = error
        self._save()

    def cancel_job(self, job_id: str) -> None:
        self.update_job(job_id, status="cancelled")

    def clear_finished(self) -> None:
        self._jobs = [j for j in self._jobs if j.status not in ("complete", "error", "cancelled")]
        self._save()

    def _save(self) -> None:
        data = {"jobs": [asdict(j) for j in self._jobs]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for item in raw.get("jobs", []):
                job = QueueJob(**item)
                if job.status == "running":
                    job.status = "queued"
                    job.progress = 0
                    job.phase = "queued"
                self._jobs.append(job)
        except (json.JSONDecodeError, TypeError, KeyError):
            self._jobs = []
```

**Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_job_queue.py -v --tb=short`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add backend/state/job_queue.py backend/tests/test_job_queue.py
git commit -m "feat: add persistent job queue with disk persistence and crash recovery"
```

---

## Task 4: Queue Worker

**Files:**
- Create: `backend/handlers/queue_worker.py`
- Test: `backend/tests/test_queue_worker.py`

**Step 1: Write the test**

`backend/tests/test_queue_worker.py`:
```python
"""Tests for the queue worker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from state.job_queue import JobQueue, QueueJob
from handlers.queue_worker import QueueWorker


class FakeJobExecutor:
    def __init__(self) -> None:
        self.executed_jobs: list[QueueJob] = []
        self.raise_on_execute: Exception | None = None

    def execute(self, job: QueueJob) -> list[str]:
        self.executed_jobs.append(job)
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        return ["/fake/output.mp4"]


def test_worker_processes_gpu_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={"prompt": "test"}, slot="gpu")

    executor = FakeJobExecutor()
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=FakeJobExecutor())
    worker.tick()

    assert len(executor.executed_jobs) == 1
    assert executor.executed_jobs[0].id == job.id
    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "complete"
    assert updated.result_paths == ["/fake/output.mp4"]


def test_worker_processes_api_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="seedance-1.5-pro", params={"prompt": "test"}, slot="api")

    api_executor = FakeJobExecutor()
    worker = QueueWorker(queue=queue, gpu_executor=FakeJobExecutor(), api_executor=api_executor)
    worker.tick()

    assert len(api_executor.executed_jobs) == 1
    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "complete"


def test_worker_handles_execution_error(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")

    executor = FakeJobExecutor()
    executor.raise_on_execute = RuntimeError("GPU exploded")
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=FakeJobExecutor())
    worker.tick()

    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "error"
    assert updated.error == "GPU exploded"


def test_worker_runs_gpu_and_api_in_parallel(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    gpu_job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    api_job = queue.submit(job_type="video", model="seedance-1.5-pro", params={}, slot="api")

    gpu_executor = FakeJobExecutor()
    api_executor = FakeJobExecutor()
    worker = QueueWorker(queue=queue, gpu_executor=gpu_executor, api_executor=api_executor)
    worker.tick()

    # Both should have been picked up in one tick
    assert len(gpu_executor.executed_jobs) == 1
    assert len(api_executor.executed_jobs) == 1


def test_worker_skips_cancelled_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue.cancel_job(job.id)

    executor = FakeJobExecutor()
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=FakeJobExecutor())
    worker.tick()

    assert len(executor.executed_jobs) == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_queue_worker.py -v --tb=short`
Expected: FAIL

**Step 3: Implement QueueWorker**

`backend/handlers/queue_worker.py`:
```python
"""Background queue worker that processes jobs from the job queue."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from state.job_queue import JobQueue, QueueJob

logger = logging.getLogger(__name__)


class JobExecutor(Protocol):
    def execute(self, job: QueueJob) -> list[str]:
        ...


class QueueWorker:
    def __init__(
        self,
        *,
        queue: JobQueue,
        gpu_executor: JobExecutor,
        api_executor: JobExecutor,
    ) -> None:
        self._queue = queue
        self._gpu_executor = gpu_executor
        self._api_executor = api_executor
        self._gpu_busy = False
        self._api_busy = False
        self._lock = threading.Lock()

    def tick(self) -> None:
        """Process one round: pick up available jobs for each free slot."""
        gpu_job: QueueJob | None = None
        api_job: QueueJob | None = None

        with self._lock:
            if not self._gpu_busy:
                gpu_job = self._queue.next_queued_for_slot("gpu")
                if gpu_job is not None:
                    self._gpu_busy = True
                    self._queue.update_job(gpu_job.id, status="running", phase="starting")

            if not self._api_busy:
                api_job = self._queue.next_queued_for_slot("api")
                if api_job is not None:
                    self._api_busy = True
                    self._queue.update_job(api_job.id, status="running", phase="starting")

        threads: list[threading.Thread] = []

        if gpu_job is not None:
            t = threading.Thread(target=self._run_job, args=(gpu_job, self._gpu_executor, "gpu"), daemon=True)
            threads.append(t)
            t.start()

        if api_job is not None:
            t = threading.Thread(target=self._run_job, args=(api_job, self._api_executor, "api"), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    def _run_job(self, job: QueueJob, executor: JobExecutor, slot: str) -> None:
        try:
            result_paths = executor.execute(job)
            self._queue.update_job(job.id, status="complete", progress=100, phase="complete", result_paths=result_paths)
        except Exception as exc:
            logger.error("Job %s failed: %s", job.id, exc)
            self._queue.update_job(job.id, status="error", error=str(exc))
        finally:
            with self._lock:
                if slot == "gpu":
                    self._gpu_busy = False
                else:
                    self._api_busy = False
```

**Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_queue_worker.py -v --tb=short`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add backend/handlers/queue_worker.py backend/tests/test_queue_worker.py
git commit -m "feat: add queue worker with dual GPU/API slot parallelism"
```

---

## Task 5: Queue API Routes

**Files:**
- Create: `backend/_routes/queue.py`
- Modify: `backend/app_factory.py`
- Modify: `backend/app_handler.py`
- Test: `backend/tests/test_queue_routes.py`

**Step 1: Write the tests**

`backend/tests/test_queue_routes.py`:
```python
"""Tests for queue API routes."""

from __future__ import annotations


def test_submit_video_job(client):
    resp = client.post("/api/queue/submit", json={
        "type": "video",
        "model": "ltx-fast",
        "params": {"prompt": "a cat", "duration": "6", "resolution": "720p", "aspectRatio": "16:9"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert "id" in data


def test_submit_image_job(client):
    resp = client.post("/api/queue/submit", json={
        "type": "image",
        "model": "z-image-turbo",
        "params": {"prompt": "a dog", "width": 1024, "height": 1024},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_get_queue_status(client):
    client.post("/api/queue/submit", json={
        "type": "video",
        "model": "ltx-fast",
        "params": {"prompt": "test"},
    })
    resp = client.get("/api/queue/status")
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "queued"


def test_cancel_job(client):
    submit_resp = client.post("/api/queue/submit", json={
        "type": "video",
        "model": "ltx-fast",
        "params": {"prompt": "test"},
    })
    job_id = submit_resp.json()["id"]
    cancel_resp = client.post(f"/api/queue/cancel/{job_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"


def test_clear_finished_jobs(client):
    submit_resp = client.post("/api/queue/submit", json={
        "type": "video",
        "model": "ltx-fast",
        "params": {"prompt": "test"},
    })
    job_id = submit_resp.json()["id"]
    client.post(f"/api/queue/cancel/{job_id}")
    client.post("/api/queue/clear")
    status_resp = client.get("/api/queue/status")
    assert len(status_resp.json()["jobs"]) == 0
```

**Step 2: Add request/response types to api_types.py**

Add to `backend/api_types.py`:
```python
class QueueSubmitRequest(BaseModel):
    type: Literal["video", "image"]
    model: str
    params: dict[str, object] = {}

class QueueJobResponse(BaseModel):
    id: str
    type: str
    model: str
    params: dict[str, object]
    status: str
    slot: str
    progress: int
    phase: str
    result_paths: list[str]
    error: str | None
    created_at: str

class QueueStatusResponse(BaseModel):
    jobs: list[QueueJobResponse]

class QueueSubmitResponse(BaseModel):
    id: str
    status: str
```

**Step 3: Implement slot assignment logic and wire queue into AppHandler**

Add to `backend/app_handler.py`:
- Import `JobQueue` from `state.job_queue`
- In `AppHandler.__init__`: create `self.job_queue = JobQueue(persistence_path=config.settings_file.parent / "job_queue.json")`
- Add a method `determine_slot(model: str) -> str` that returns:
  - `"api"` if model is `"seedance-1.5-pro"` or `"nano-banana-2"`
  - `"api"` if `self.config.force_api_generations` is True
  - `"gpu"` otherwise

**Step 4: Create the route file**

`backend/_routes/queue.py`:
```python
"""Route handlers for /api/queue/*."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import QueueSubmitRequest, QueueSubmitResponse, QueueStatusResponse, QueueJobResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api/queue", tags=["queue"])


@router.post("/submit", response_model=QueueSubmitResponse)
def route_queue_submit(
    req: QueueSubmitRequest,
    handler: AppHandler = Depends(get_state_service),
) -> QueueSubmitResponse:
    slot = handler.determine_slot(req.model)
    job = handler.job_queue.submit(
        job_type=req.type,
        model=req.model,
        params=dict(req.params),
        slot=slot,
    )
    return QueueSubmitResponse(id=job.id, status=job.status)


@router.get("/status", response_model=QueueStatusResponse)
def route_queue_status(
    handler: AppHandler = Depends(get_state_service),
) -> QueueStatusResponse:
    jobs = handler.job_queue.get_all_jobs()
    return QueueStatusResponse(jobs=[
        QueueJobResponse(
            id=j.id, type=j.type, model=j.model, params=dict(j.params),
            status=j.status, slot=j.slot, progress=j.progress, phase=j.phase,
            result_paths=j.result_paths, error=j.error, created_at=j.created_at,
        )
        for j in jobs
    ])


@router.post("/cancel/{job_id}")
def route_queue_cancel(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> QueueSubmitResponse:
    handler.job_queue.cancel_job(job_id)
    job = handler.job_queue.get_job(job_id)
    status = job.status if job else "not_found"
    return QueueSubmitResponse(id=job_id, status=status)


@router.post("/clear")
def route_queue_clear(
    handler: AppHandler = Depends(get_state_service),
) -> QueueStatusResponse:
    handler.job_queue.clear_finished()
    return route_queue_status(handler)
```

**Step 5: Register router in app_factory.py**

Add to `backend/app_factory.py`:
- Import: `from _routes.queue import router as queue_router`
- Add: `app.include_router(queue_router)`

**Step 6: Run tests**

Run: `cd backend && uv run pytest tests/test_queue_routes.py -v --tb=short`
Expected: All 5 tests PASS

Run: `cd backend && uv run pytest -v --tb=short`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add backend/_routes/queue.py backend/api_types.py backend/app_handler.py
git add backend/app_factory.py backend/tests/test_queue_routes.py
git commit -m "feat: add queue API routes (submit, status, cancel, clear)"
```

---

## Task 6: Frontend — Settings (video_model)

**Files:**
- Modify: `frontend/contexts/AppSettingsContext.tsx`
- Modify: `frontend/components/SettingsModal.tsx`

**Step 1: Add videoModel to AppSettingsContext**

In `frontend/contexts/AppSettingsContext.tsx`:
- Add to `AppSettings` interface: `videoModel: string`
- Add to `DEFAULT_APP_SETTINGS`: `videoModel: 'ltx-fast'`
- Add to `normalizeAppSettings`: `videoModel: data.videoModel ?? DEFAULT_APP_SETTINGS.videoModel`

**Step 2: Add Video Model dropdown to SettingsModal**

In `frontend/components/SettingsModal.tsx`, add a "Video Model" select with options:
- `ltx-fast` — "LTX Fast"
- `seedance-1.5-pro` — "Seedance 1.5 Pro"

Bound to `settings.videoModel`, save via `updateSettings({ videoModel: value })`.

**Step 3: Commit**

```bash
git add frontend/contexts/AppSettingsContext.tsx frontend/components/SettingsModal.tsx
git commit -m "feat: add video model selector (LTX Fast | Seedance 1.5 Pro) to settings"
```

---

## Task 7: Frontend — Queue UI + use-generation Refactor

**Files:**
- Modify: `frontend/hooks/use-generation.ts`
- Modify: relevant view component that shows generation results

**Step 1: Update use-generation.ts to submit to queue**

Replace the direct `POST /api/generate` and `POST /api/generate-image` calls with `POST /api/queue/submit`. Replace progress polling from `/api/generation/progress` to `/api/queue/status`.

The hook should:
1. Submit job to `/api/queue/submit` with `{type, model, params}`
2. Poll `/api/queue/status` every 500ms
3. Track all jobs (not just the latest) — expose `jobs` array
4. Keep the existing `generate()` / `generateImage()` API signatures for now, just route through queue

**Step 2: Show queue status in results area**

Update the results area component to render `jobs` from use-generation:
- Each job: status badge, prompt preview, progress bar (running), thumbnail/link (complete), error message (error)
- Multiple jobs can show simultaneously
- Generate button re-enables immediately after submission

**Step 3: Commit**

```bash
git add frontend/hooks/use-generation.ts frontend/components/ frontend/views/
git commit -m "feat: frontend queue UI — submit to queue, poll status, show all jobs"
```

---

## Task 8: Run Full Test Suite + Type Checks

**Step 1: Run backend tests**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: All tests PASS

**Step 2: Run Python type checks**

Run: `cd backend && uv run pyright`
Expected: No errors

**Step 3: Run TypeScript type checks**

Run: `npx pnpm typecheck:ts`
Expected: No errors

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: all tests and type checks passing for queue + seedance"
```

---

## Slot Assignment Summary

| Model | Condition | Slot |
|-------|-----------|------|
| `ltx-fast` | local GPU available (`force_api=false`) | `gpu` |
| `ltx-fast` | `force_api=true` or no GPU | `api` |
| `seedance-1.5-pro` | always | `api` |
| `z-image-turbo` | local GPU available (`force_api=false`) | `gpu` |
| `z-image-turbo` | `force_api=true` or no GPU | `api` |
| `nano-banana-2` | always | `api` |
