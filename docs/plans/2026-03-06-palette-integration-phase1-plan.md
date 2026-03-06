# Director's Desktop — Palette Integration Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Palette auth connection (API key to start), sync status route, and first/last frame video generation — the two highest-impact features that make Desktop feel integrated and powerful.

**Architecture:** Desktop's FastAPI backend gets new `/api/sync/*` routes for Palette connection and a `PaletteSyncClient` service for calling Palette's API. The video generation pipeline gets `lastFramePath` support for both LTX local (via `frame_idx`) and Seedance (via Replicate `last_frame` param). Frontend gets first/last frame image slots with paste/drop/browse support.

**Tech Stack:** Python 3.12+ (FastAPI, Pydantic, pytest), TypeScript (React 18, Electron), Supabase (Palette backend)

---

## Part A: Palette Sync — API Key Auth + Status

### Task 1: Palette API Key Setting

**Files:**
- Modify: `backend/state/app_settings.py`
- Modify: `backend/handlers/settings_handler.py:66`
- Modify: `frontend/contexts/AppSettingsContext.tsx`
- Modify: `settings.json`
- Test: `backend/tests/test_settings.py`

**Step 1: Write the failing test**

Add to `backend/tests/test_settings.py`:

```python
def test_palette_api_key_roundtrip(client, default_app_settings):
    """Palette API key can be saved and is masked in responses."""
    resp = client.patch("/api/settings", json={"paletteApiKey": "dp_test_key_123"})
    assert resp.status_code == 200

    resp = client.get("/api/settings")
    data = resp.json()
    assert data["hasPaletteApiKey"] is True
    assert "dp_test_key_123" not in resp.text
```

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_settings.py::test_palette_api_key_roundtrip -v --tb=short`
Expected: FAIL — `hasPaletteApiKey` not in response

**Step 3: Implement**

In `backend/state/app_settings.py`, add to `AppSettings` class:
```python
palette_api_key: str = ""
```

In `backend/state/app_settings.py`, add to `SettingsResponse`:
```python
has_palette_api_key: bool = False
```

In `backend/state/app_settings.py`, in `to_settings_response()` method, add `palette_api_key` to the popped keys list and set `has_palette_api_key`.

In `backend/handlers/settings_handler.py`, add `"palette_api_key"` to the key fields list (around line 66).

In `frontend/contexts/AppSettingsContext.tsx`, add:
```typescript
hasPaletteApiKey: boolean  // in interface
paletteApiKey: ''          // in defaults
```

In `settings.json`, add:
```json
"palette_api_key": ""
```

**Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_settings.py::test_palette_api_key_roundtrip -v --tb=short`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/state/app_settings.py backend/handlers/settings_handler.py frontend/contexts/AppSettingsContext.tsx settings.json backend/tests/test_settings.py
git commit -m "feat: add palette_api_key to app settings"
```

---

### Task 2: PaletteSyncClient Service Protocol

**Files:**
- Create: `backend/services/palette_sync_client/palette_sync_client.py`
- Create: `backend/services/palette_sync_client/__init__.py`
- Modify: `backend/services/interfaces.py`

**Step 1: Create the Protocol**

```python
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
```

**Step 2: Create `__init__.py`**

```python
from services.palette_sync_client.palette_sync_client import PaletteSyncClient

__all__ = ["PaletteSyncClient"]
```

**Step 3: Add to interfaces.py**

Add import and `__all__` entry for `PaletteSyncClient`.

**Step 4: Commit**

```bash
git add backend/services/palette_sync_client/ backend/services/interfaces.py
git commit -m "feat: add PaletteSyncClient protocol"
```

---

### Task 3: Fake PaletteSyncClient + Test Infrastructure

**Files:**
- Modify: `backend/tests/fakes/services.py`
- Modify: `backend/tests/conftest.py`

**Step 1: Add FakePaletteSyncClient to fakes**

```python
class FakePaletteSyncClient:
    def __init__(self) -> None:
        self.validate_calls: list[str] = []
        self.credits_calls: list[str] = []
        self.raise_on_validate: Exception | None = None
        self.user_info: dict[str, Any] = {"id": "user-123", "email": "test@example.com", "name": "Test User"}
        self.credits_info: dict[str, Any] = {"balance": 5000, "currency": "credits"}

    def validate_connection(self, *, api_key: str) -> dict[str, Any]:
        self.validate_calls.append(api_key)
        if self.raise_on_validate is not None:
            raise self.raise_on_validate
        return self.user_info

    def get_credits(self, *, api_key: str) -> dict[str, Any]:
        self.credits_calls.append(api_key)
        return self.credits_info
```

Add to `FakeServices` dataclass:
```python
palette_sync_client: FakePaletteSyncClient = field(default_factory=FakePaletteSyncClient)
```

Update `conftest.py` ServiceBundle construction to include `palette_sync_client=fake_services.palette_sync_client`.

**Step 2: Commit**

```bash
git add backend/tests/fakes/services.py backend/tests/conftest.py
git commit -m "feat: add FakePaletteSyncClient test double"
```

---

### Task 4: Sync Routes — Connect + Status

**Files:**
- Create: `backend/_routes/sync.py`
- Create: `backend/handlers/sync_handler.py`
- Modify: `backend/app_factory.py`
- Modify: `backend/app_handler.py`
- Create: `backend/tests/test_sync.py`

**Step 1: Write the failing tests**

Create `backend/tests/test_sync.py`:

```python
"""Tests for Palette sync routes."""
from __future__ import annotations

import pytest


class TestSyncStatus:
    def test_disconnected_by_default(self, client):
        resp = client.get("/api/sync/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False
        assert data["user"] is None

    def test_connected_after_setting_api_key(self, client):
        client.patch("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        resp = client.get("/api/sync/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert data["user"]["email"] == "test@example.com"

    def test_connection_fails_with_invalid_key(self, client, fake_services):
        fake_services.palette_sync_client.raise_on_validate = RuntimeError("Invalid API key")
        client.patch("/api/settings", json={"paletteApiKey": "dp_bad_key"})
        resp = client.get("/api/sync/status")
        data = resp.json()
        assert data["connected"] is False


class TestSyncCredits:
    def test_credits_when_connected(self, client):
        client.patch("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        resp = client.get("/api/sync/credits")
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"] == 5000

    def test_credits_when_disconnected(self, client):
        resp = client.get("/api/sync/credits")
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"] is None
        assert data["connected"] is False
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_sync.py -v --tb=short`
Expected: FAIL — module not found

**Step 3: Implement sync_handler.py**

```python
"""Handler for Palette sync operations."""
from __future__ import annotations

from typing import Any

from services.palette_sync_client.palette_sync_client import PaletteSyncClient
from state.app_state_types import AppState


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
```

**Step 4: Implement sync routes**

Create `backend/_routes/sync.py`:
```python
"""Palette sync routes."""
from __future__ import annotations

from fastapi import APIRouter
from state.deps import get_handler

router = APIRouter(prefix="/api/sync", tags=["sync"])


@router.get("/status")
def sync_status():
    return get_handler().sync.get_status()


@router.get("/credits")
def sync_credits():
    return get_handler().sync.get_credits()
```

**Step 5: Wire into AppHandler and app_factory**

In `app_handler.py`: Add `PaletteSyncClient` to imports, `__init__` params, and create `self.sync = SyncHandler(...)`.

In `app_factory.py`: Import and register `sync_router`.

In `ServiceBundle`: Add `palette_sync_client` field.

**Step 6: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_sync.py -v --tb=short`
Expected: ALL PASS

**Step 7: Run full test suite**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: ALL PASS (no regressions)

**Step 8: Commit**

```bash
git add backend/_routes/sync.py backend/handlers/sync_handler.py backend/app_factory.py backend/app_handler.py backend/tests/test_sync.py
git commit -m "feat: add /api/sync/status and /api/sync/credits routes"
```

---

## Part B: First Frame + Last Frame Video Generation

### Task 5: Add lastFramePath to API Types

**Files:**
- Modify: `backend/api_types.py`
- Test: `backend/tests/test_generation.py`

**Step 1: Write the failing test**

Add to `backend/tests/test_generation.py`:

```python
def test_generate_video_request_accepts_last_frame_path():
    from api_types import GenerateVideoRequest
    req = GenerateVideoRequest(prompt="test", lastFramePath="/path/to/last.png")
    assert req.lastFramePath == "/path/to/last.png"

def test_generate_video_request_last_frame_defaults_none():
    from api_types import GenerateVideoRequest
    req = GenerateVideoRequest(prompt="test")
    assert req.lastFramePath is None
```

**Step 2: Run to verify failure**

Run: `cd backend && uv run pytest tests/test_generation.py::test_generate_video_request_accepts_last_frame_path -v --tb=short`
Expected: FAIL

**Step 3: Add field to GenerateVideoRequest**

In `backend/api_types.py`, add to `GenerateVideoRequest`:
```python
lastFramePath: str | None = None
```

**Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/test_generation.py::test_generate_video_request_accepts_last_frame_path tests/test_generation.py::test_generate_video_request_last_frame_defaults_none -v --tb=short`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/api_types.py backend/tests/test_generation.py
git commit -m "feat: add lastFramePath to GenerateVideoRequest"
```

---

### Task 6: Wire Last Frame into LTX Local Pipeline

**Files:**
- Modify: `backend/handlers/video_generation_handler.py`
- Test: `backend/tests/test_generation.py`

**Step 1: Write the failing test**

```python
def test_local_generation_with_last_frame(client, test_state, create_fake_model_files, make_test_image, tmp_path):
    """Last frame image should be passed as ImageConditioningInput with frame_idx=last."""
    create_fake_model_files()
    # Force local generation
    test_state.state.app_settings.ltx_api_key = ""

    # Create a fake last-frame image
    img_buf = make_test_image(512, 512)
    img_path = tmp_path / "last_frame.png"
    img_path.write_bytes(img_buf.read())

    resp = client.post("/api/generate-video", json={
        "prompt": "A cat walks across the room",
        "resolution": "540p",
        "model": "fast",
        "duration": "2",
        "fps": "24",
        "lastFramePath": str(img_path),
    })
    assert resp.status_code == 200

    # Verify the pipeline received last-frame conditioning
    calls = test_state.fast_video_pipeline_class._singleton.generate_calls
    assert len(calls) == 1
    images = calls[0]["images"]
    # Should have at least one image with frame_idx > 0
    last_frame_images = [img for img in images if img.frame_idx > 0]
    assert len(last_frame_images) == 1
```

**Step 2: Run to verify failure**

Run: `cd backend && uv run pytest tests/test_generation.py::test_local_generation_with_last_frame -v --tb=short`
Expected: FAIL — lastFramePath not handled

**Step 3: Implement in video_generation_handler.py**

In the `_generate_local()` method, after the existing first-frame image conditioning block, add:

```python
if request.lastFramePath:
    last_frame_image = Image.open(request.lastFramePath).convert("RGB")
    # num_frames - 1 is the last frame index
    images.append(ImageConditioningInput(
        path=request.lastFramePath,
        frame_idx=num_frames - 1,
        strength=1.0,
    ))
```

**Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/test_generation.py::test_local_generation_with_last_frame -v --tb=short`
Expected: PASS

**Step 5: Run full suite**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add backend/handlers/video_generation_handler.py backend/tests/test_generation.py
git commit -m "feat: wire lastFramePath into LTX local pipeline via frame_idx"
```

---

### Task 7: Wire Last Frame into Seedance (Replicate)

**Files:**
- Modify: `backend/services/video_api_client/video_api_client.py`
- Modify: `backend/services/video_api_client/replicate_video_client_impl.py`
- Modify: `backend/tests/fakes/services.py`
- Test: `backend/tests/test_video_api_client.py`

**Step 1: Write the failing test**

Add to `backend/tests/test_video_api_client.py`:

```python
def test_seedance_with_last_frame():
    """Seedance should pass last_frame in input payload."""
    http = FakeHTTPClient()
    # Queue prediction response (sync success)
    http.queue("post", FakeResponse(
        status_code=200,
        json_payload={"status": "succeeded", "output": "https://example.com/video.mp4"},
    ))
    # Queue video download
    http.queue("get", FakeResponse(status_code=200, content=b"fake-video"))

    client = ReplicateVideoClientImpl(http=http)
    result = client.generate_text_to_video(
        api_key="test-key",
        model="seedance-1.5-pro",
        prompt="A cat",
        duration=5,
        resolution="720p",
        aspect_ratio="16:9",
        generate_audio=False,
        last_frame_path="/tmp/last.png",
    )

    # Verify the POST payload included last_frame
    post_call = http.calls[0]
    payload = post_call.json_payload
    assert "last_frame" in payload["input"]
```

**Step 2: Update Protocol to accept last_frame_path**

In `video_api_client.py`:
```python
def generate_text_to_video(
    self, *, api_key: str, model: str, prompt: str,
    duration: int, resolution: str, aspect_ratio: str,
    generate_audio: bool, last_frame_path: str | None = None,
) -> bytes: ...
```

**Step 3: Update ReplicateVideoClientImpl**

Add `last_frame_path` parameter. For seedance, if provided, read the image, base64-encode it, and include as `last_frame` in the input payload (or as a data URI).

**Step 4: Update FakeVideoAPIClient**

Add `last_frame_path` parameter to `generate_text_to_video`.

**Step 5: Run tests**

Run: `cd backend && uv run pytest tests/test_video_api_client.py -v --tb=short`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add backend/services/video_api_client/ backend/tests/fakes/services.py backend/tests/test_video_api_client.py
git commit -m "feat: add last_frame_path support to Seedance via Replicate"
```

---

### Task 8: Queue Submit with Last Frame

**Files:**
- Modify: `backend/app_handler.py` (determine_slot and queue submit logic)
- Modify: `backend/_routes/queue.py`
- Test: `backend/tests/test_queue_routes.py`

**Step 1: Write the failing test**

```python
def test_queue_submit_video_with_last_frame(client):
    resp = client.post("/api/queue/submit", json={
        "type": "video",
        "prompt": "A cat walks",
        "model": "ltx-fast",
        "params": {"lastFramePath": "/tmp/last.png", "resolution": "540p"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"

    # Verify params contain lastFramePath
    status = client.get("/api/queue/status").json()
    job = status["jobs"][0]
    assert job["params"]["lastFramePath"] == "/tmp/last.png"
```

**Step 2: Implement** — The queue already passes arbitrary `params` through, so this should mostly work. Verify the queue worker passes `lastFramePath` to the video generation handler.

**Step 3: Run tests**

Run: `cd backend && uv run pytest tests/test_queue_routes.py -v --tb=short`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add backend/ backend/tests/test_queue_routes.py
git commit -m "feat: queue submit supports lastFramePath param"
```

---

### Task 9: Frontend — First/Last Frame Image Slots

**Files:**
- Create: `frontend/components/FrameSlot.tsx`
- Modify: `frontend/views/Playground.tsx`
- Modify: `frontend/hooks/use-generation.ts`

**Step 1: Create FrameSlot component**

A reusable image slot that supports:
- Paste (Ctrl+V / Cmd+V)
- Drag & drop
- Click to browse
- Thumbnail preview with X to clear
- Label ("First Frame" / "Last Frame")

```typescript
interface FrameSlotProps {
  label: string
  imageUrl: string | null
  onImageSet: (url: string | null, path: string | null) => void
}
```

**Step 2: Add to Playground**

Add two FrameSlot components above the prompt area:
```tsx
<div className="grid grid-cols-2 gap-3">
  <FrameSlot label="First Frame" imageUrl={firstFrameUrl} onImageSet={setFirstFrame} />
  <FrameSlot label="Last Frame" imageUrl={lastFrameUrl} onImageSet={setLastFrame} />
</div>
```

**Step 3: Wire into generation hook**

Update `use-generation.ts` `generate()` to accept `lastFramePath` and pass it in the queue submit request params.

**Step 4: Verify TypeScript compiles**

Run: `cd D:/git/directors-desktop && npx tsc --noEmit`
Expected: clean

**Step 5: Commit**

```bash
git add frontend/components/FrameSlot.tsx frontend/views/Playground.tsx frontend/hooks/use-generation.ts
git commit -m "feat: add first/last frame image slots to Playground UI"
```

---

### Task 10: Frontend — Palette Connection UI in Settings

**Files:**
- Modify: `frontend/components/SettingsModal.tsx`
- Modify: `frontend/contexts/AppSettingsContext.tsx`

**Step 1: Add Palette API Key input to Settings**

In SettingsModal, add a new "Director's Palette" section:
- API Key input field (password type, with show/hide toggle)
- Connection status indicator (green dot = connected, red = disconnected)
- Credits display when connected
- "Get API Key" link button

**Step 2: Add sync status polling**

In AppSettingsContext, add:
- `paletteConnected: boolean`
- `paletteUser: { email: string; name: string } | null`
- `paletteCredits: number | null`
- Poll `/api/sync/status` and `/api/sync/credits` every 60 seconds when key is set

**Step 3: Verify TypeScript compiles**

Run: `cd D:/git/directors-desktop && npx tsc --noEmit`
Expected: clean

**Step 4: Commit**

```bash
git add frontend/components/SettingsModal.tsx frontend/contexts/AppSettingsContext.tsx
git commit -m "feat: add Palette connection UI to Settings modal"
```

---

### Task 11: Image Variations Slider

**Files:**
- Modify: `frontend/components/SettingsPanel.tsx`
- No backend changes needed (already supports 1-12 variations)

**Step 1: Add variations slider**

In SettingsPanel, when mode is `text-to-image`, add:
```tsx
<div className="space-y-1">
  <label className="text-xs text-zinc-500">Variations</label>
  <input type="range" min={1} max={12} value={settings.variations || 1}
    onChange={(e) => onSettingsChange({...settings, variations: parseInt(e.target.value)})} />
  <span className="text-xs text-zinc-400">{settings.variations || 1}</span>
</div>
```

**Step 2: Verify TypeScript compiles + commit**

```bash
git add frontend/components/SettingsPanel.tsx
git commit -m "feat: expose image variations slider (1-12)"
```

---

### Task 12: Social Media Aspect Ratio Labels

**Files:**
- Modify: `frontend/components/SettingsPanel.tsx`

**Step 1: Update aspect ratio labels**

Change the video aspect ratio options:
```tsx
<option value="16:9">16:9 — YouTube / Landscape</option>
<option value="9:16">9:16 — TikTok / Reels / Shorts</option>
```

For image aspect ratios, add labels and the new 4:5 option:
```tsx
<option value="1:1">1:1 — Square</option>
<option value="16:9">16:9 — YouTube</option>
<option value="9:16">9:16 — TikTok / Reels</option>
<option value="4:3">4:3 — Standard</option>
<option value="3:4">3:4 — Portrait</option>
<option value="4:5">4:5 — Instagram Post</option>
<option value="21:9">21:9 — Cinematic</option>
```

**Step 2: Commit**

```bash
git add frontend/components/SettingsPanel.tsx
git commit -m "feat: add social media labels to aspect ratio presets"
```

---

### Task 13: Prompt Enhancement Button

**Files:**
- Create: `backend/_routes/enhance_prompt.py`
- Create: `backend/handlers/enhance_prompt_handler.py`
- Modify: `backend/app_factory.py`
- Modify: `backend/app_handler.py`
- Create: `backend/tests/test_enhance_prompt.py`
- Modify: `frontend/views/Playground.tsx`

**Step 1: Write the failing test**

```python
def test_enhance_prompt_returns_enhanced_text(client, fake_services):
    """Enhance prompt route should return an enhanced version of the input."""
    # Configure Gemini key
    client.patch("/api/settings", json={"geminiApiKey": "test-gemini-key"})

    # Queue a fake Gemini response
    from tests.fakes.services import FakeResponse
    fake_services.http.queue("post", FakeResponse(
        status_code=200,
        json_payload={
            "candidates": [{"content": {"parts": [{"text": "A cinematic shot of a majestic cat walking gracefully across a sun-drenched room, golden hour lighting, shallow depth of field"}]}}]
        },
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "cat walking in room",
        "mode": "text-to-video",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "enhancedPrompt" in data
    assert len(data["enhancedPrompt"]) > len("cat walking in room")
```

**Step 2: Implement handler + route**

The handler calls Gemini API (same pattern as `suggest_gap_prompt_handler.py`) with a system prompt asking it to enhance the user's rough prompt into a detailed cinematic description.

**Step 3: Add sparkle button to Playground prompt area**

Next to the prompt textarea, add a small button with a Sparkles icon that calls `/api/enhance-prompt` and replaces the prompt text.

**Step 4: Run tests + commit**

```bash
git add backend/_routes/enhance_prompt.py backend/handlers/enhance_prompt_handler.py backend/app_factory.py backend/app_handler.py backend/tests/test_enhance_prompt.py frontend/views/Playground.tsx
git commit -m "feat: add prompt enhancement button (Gemini-powered)"
```

---

### Task 14: Full Test Suite Verification

**Step 1: Run all backend tests**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: ALL PASS

**Step 2: Run TypeScript type check**

Run: `cd D:/git/directors-desktop && npx tsc --noEmit`
Expected: clean

**Step 3: Run Python type check**

Run: `cd backend && uv run pyright`
Expected: 0 errors

**Step 4: Manual smoke test**

Start the app: `npx pnpm dev`
- Verify Playground shows First Frame / Last Frame slots
- Verify paste (Ctrl+V) works in frame slots
- Verify Settings shows Palette API Key section
- Verify image variations slider appears in text-to-image mode
- Verify aspect ratio labels show platform names
- Verify prompt enhancement button appears

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: Phase 1 complete — Palette auth + first/last frame + generation upgrades"
```
