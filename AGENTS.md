# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LTX Desktop is an Electron app for AI video generation using LTX models. Three-layer architecture:

- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind CSS renderer
- **Electron** (`electron/`): Main process managing app lifecycle, IPC, Python backend process, ffmpeg export
- **Backend** (`backend/`): Python FastAPI server (default port 8010) handling ML model orchestration and generation

## Common Commands

| Command | Purpose |
|---|---|
| `pnpm dev` | Start dev server (Vite + Electron + Python backend) |
| `pnpm dev:debug` | Dev with Electron inspector + Python debugpy |
| `pnpm typecheck` | Run TypeScript (`tsc --noEmit`) and Python (`pyright`) type checks |
| `pnpm typecheck:ts` | TypeScript only |
| `pnpm typecheck:py` | Python pyright only |
| `pnpm backend:test` | Run Python pytest tests |
| `pnpm build:frontend` | Vite frontend build only |
| `pnpm build:mac` / `pnpm build:win` | Full platform builds |
| `pnpm setup:dev:mac` / `pnpm setup:dev:win` | One-time dev environment setup |

Run a single backend test: `cd backend && uv run pytest tests/test_generation.py -v --tb=short`

## CI Checks

PRs must pass: `pnpm typecheck` + `pnpm backend:test` + frontend Vite build.

## Frontend Architecture

- **Path alias**: `@/*` maps to `frontend/*`
- **State management**: React contexts only (`ProjectContext`, `AppSettingsContext`, `KeyboardShortcutsContext`) — no Redux/Zustand
- **Routing**: View-based via `ProjectContext` with views: `home`, `project`, `playground`
- **IPC bridge**: All Electron communication through `window.electronAPI` (defined in `electron/preload.ts`)
- **Backend calls**: Frontend calls the backend at `http://localhost:8010` (configurable port) directly
- **Styling**: Tailwind with custom semantic color tokens via CSS variables; utilities from `class-variance-authority` + `clsx` + `tailwind-merge`
- **No frontend tests** currently exist

## Backend Architecture

Request flow: `_routes/* (thin) → AppHandler → handlers/* (logic) → services/* (side effects) + state/* (mutations)`

Key patterns:
- **Routes** (`_routes/`): Thin plumbing only — parse input, call handler, return typed output. No business logic.
- **AppHandler** (`app_handler.py`): Single composition root owning all sub-handlers, state, and lock
- **State** (`state/`): Centralized `AppState` using discriminated union types for state machines (e.g., `GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled`)
- **Services** (`services/`): Protocol interfaces with real implementations and fake test implementations. The test boundary for heavy side effects (GPU, network).
- **Concurrency**: Thread pool with shared `RLock`. Pattern: lock→read/validate→unlock→heavy work→lock→write. Never hold lock during heavy compute/IO.
- **Exception handling**: Boundary-owned traceback policy. Handlers raise `HTTPError` with `from exc` chaining; `app_factory.py` owns logging. Don't `logger.exception()` then rethrow.
- **Naming**: `*Payload` for DTOs/TypedDicts, `*Like` for structural wrappers, `Fake*` for test implementations

### Backend Testing

- Integration-first using Starlette `TestClient` against real FastAPI app
- **No mocks**: `test_no_mock_usage.py` enforces no `unittest.mock`. Swap services via `ServiceBundle` fakes only.
- Fakes live in `tests/fakes/`; `conftest.py` wires fresh `AppHandler` per test
- Pyright strict mode is also enforced as a test (`test_pyright.py`)

### Adding a Backend Feature

1. Define request/response models in `api_types.py`
2. Add endpoint in `_routes/<domain>.py` delegating to handler
3. Implement logic in `handlers/<domain>_handler.py` with lock-aware state transitions
4. If new heavy side effect needed, add service in `services/` with Protocol + real + fake implementations
5. Add integration test in `tests/` using fake services

## TypeScript Config

- Strict mode with `noUnusedLocals`, `noUnusedParameters`
- Frontend: ES2020 target, React JSX
- Electron main process: ESNext, compiled to `dist-electron/`
- Preload script must be CommonJS

## Python Config

- Python 3.13+ (per `.python-version`), managed with `uv`
- Pyright strict mode (`backend/pyrightconfig.json`)
- Dependencies in `backend/pyproject.toml`

## GGUF (experimental)

- **Settings**: Use GGUF models, quantization (Q2_K–Q8_0), and optional ComfyUI models path are in app settings (Settings → Inference). Restart required for changes.
- **ENV overrides**: `LTX_USE_GGUF=1`, `LTX_GGUF_QUANTIZATION`, `LTX_COMFYUI_MODELS_PATH` can override at startup.
- **ComfyUI path**: On macOS, empty ComfyUI path defaults to `~/Documents/ComfyUI/models` when present. When set, checkpoint/upsampler/text_encoder resolve under that base (unet/, latent_upscale_models/, text_encoders/). ZIT stays under app models dir.
- **Behavior**: GGUF mode uses Unsloth LTX-2.3-GGUF; in-app inference is not implemented (stub raises on generate). Use ComfyUI-GGUF with the downloaded or shared models.
- **Plan**: `backend/docs/GGUF_INTEGRATION_PLAN.md`

## Key File Locations

- Backend architecture doc: `backend/architecture.md`
- Default app settings schema: `settings.json`
- Electron builder config: `electron-builder.yml`
- Video editor (largest frontend file): `frontend/views/VideoEditor.tsx`
- Project types: `frontend/types/project.ts`
