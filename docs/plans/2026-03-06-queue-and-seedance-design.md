# Design: Generation Queue + Seedance 1.5 Pro

## Summary

Add a persistent job queue to LTX Desktop so users can submit multiple generation requests that process sequentially. Add Seedance 1.5 Pro as a video model option via Replicate API. Allow local GPU jobs and API jobs to run in parallel.

## Decisions

| Decision | Choice |
|----------|--------|
| Queue type | Simple sequential, fire-and-forget |
| Persistence | Disk-persisted, survives restarts |
| Seedance UI placement | Same video generation view, model dropdown |
| Model selection scope | Global setting |
| Queue UI location | Existing generation results area |
| Parallelism | GPU slot + API slot can run concurrently |
| Seedance routing | Always API (Replicate), regardless of force_api |

## Architecture

```
Frontend queue UI (existing results area)
    | POST /api/queue/submit
Backend JobQueue (persistent JSON file)
    | QueueWorker thread (picks next job)
    |-- GPU slot (LTX video, ZIT images)
    +-- API slot (Seedance, Replicate images)  <-- parallel
    | results written to outputs/
Frontend polls GET /api/queue/status
```

## Backend Changes

### Job Queue State (`backend/state/job_queue.py`)

Jobs persisted to JSON file in app data directory alongside settings.json.

```python
@dataclass
class QueueJob:
    id: str
    type: Literal["video", "image"]
    model: str
    params: dict[str, Any]
    status: Literal["queued", "running", "complete", "error", "cancelled"]
    slot: Literal["gpu", "api"]
    progress: int  # 0-100
    phase: str  # "queued", "loading_model", "inference", etc.
    result_paths: list[str]
    error: str | None
    created_at: str  # ISO 8601
```

On startup, any `running` jobs reset to `queued` (crash recovery).

### Queue Worker (`backend/handlers/queue_worker.py`)

Background thread started on app boot. Two concurrent slots:

- **GPU slot**: local LTX video, local ZIT image generation
- **API slot**: Seedance (Replicate), Replicate image models, LTX API video

Worker loop:
1. Check if GPU slot is free -> pick next queued GPU-type job
2. Check if API slot is free -> pick next queued API-type job
3. Both can run simultaneously
4. Sleep 500ms between checks

Reuses existing handler logic internally (VideoGenerationHandler, ImageGenerationHandler).

### Slot Assignment

| Model | Slot |
|-------|------|
| ltx-fast (local GPU available) | gpu |
| ltx-fast (force_api or no GPU) | api |
| seedance-1.5-pro | api (always) |
| z-image-turbo (local GPU available) | gpu |
| z-image-turbo (force_api or no GPU) | api |
| nano-banana-2 | api (always) |

### New Routes (`backend/_routes/queue.py`)

- `POST /api/queue/submit` - Add job, returns `{id, status}`
- `GET /api/queue/status` - Returns all jobs with progress
- `POST /api/queue/cancel/{job_id}` - Cancel specific job
- `POST /api/queue/clear` - Remove completed/errored jobs

### Seedance Video Client (`backend/services/video_api_client/`)

New service directory following existing patterns:

**Protocol** (`video_api_client.py`):
```python
class VideoAPIClient(Protocol):
    def generate_text_to_video(
        self, *, api_key: str, model: str,
        prompt: str, duration: int, resolution: str,
        aspect_ratio: str, generate_audio: bool,
    ) -> bytes: ...
```

**Implementation** (`replicate_video_client_impl.py`):
- Model routing: `seedance-1.5-pro` -> `bytedance/seedance-1.5-pro`
- Input: `{prompt, duration, resolution, ratio, generate_audio}`
- Duration: 4-12 seconds
- Resolution: 480p, 720p
- Aspect ratios: 16:9, 9:16, 1:1, 4:3, 3:4, 21:9
- Same Prefer: wait + polling pattern as image client
- Output: video bytes (mp4)

### Settings Changes (`backend/state/app_settings.py`)

Add:
- `video_model: str = "ltx-fast"` (choices: `"ltx-fast"`, `"seedance-1.5-pro"`)

SettingsResponse:
- Add `video_model: str`

### AppHandler Wiring

- Add `video_api_client: VideoAPIClient` to ServiceBundle
- Wire `ReplicateVideoClientImpl` in `build_default_service_bundle`
- Pass to QueueWorker

## Frontend Changes

### Queue in Results Area

The existing generation results area becomes a job list:
- Each job: status badge, prompt preview, progress bar (if running), result thumbnail (if complete)
- Generate button submits to queue and immediately re-enables
- Two progress bars can show simultaneously (GPU + API)
- Completed jobs show clickable results (same as current behavior)

### Settings Modal

- Add **Video Model** dropdown: LTX Fast | Seedance 1.5 Pro
- When Seedance selected, resolution options change to 480p/720p
- Duration range changes to 4-12s for Seedance

### AppSettingsContext

- Add `videoModel: string` (default: `"ltx-fast"`)

### use-generation.ts

- POST to `/api/queue/submit` instead of `/api/generate` or `/api/generate-image`
- Poll `/api/queue/status` for all job progress
- Handle multiple concurrent results

## Persistence Format

```json
{
  "jobs": [
    {
      "id": "a1b2c3",
      "type": "video",
      "model": "seedance-1.5-pro",
      "params": {"prompt": "...", "duration": 8, "resolution": "720p", "aspect_ratio": "16:9", "generate_audio": true},
      "status": "complete",
      "slot": "api",
      "progress": 100,
      "phase": "complete",
      "result_paths": ["/path/to/output.mp4"],
      "error": null,
      "created_at": "2026-03-06T02:30:00Z"
    }
  ]
}
```

## Testing Strategy

- Fake queue worker for unit tests (no real GPU/API calls)
- Integration tests: submit job, verify status transitions
- Test parallel slot execution (GPU + API simultaneously)
- Test persistence: write queue, reload, verify recovery
- Test Seedance client with FakeHTTPClient (same pattern as image client tests)
- Test cancel mid-queue, cancel running job
