# Director's Desktop + Director's Palette Integration Design

## Goal

Transform Directors Desktop from a standalone local AI generation app into the **local creative powerhouse** paired with Director's Palette (web/mobile). Palette is where you plan, organize, and manage your creative library in the cloud. Desktop is where you render, edit, and experiment with unlimited local power. They share auth, gallery, library, and characters.

## Architecture

```
Director's Palette (Web/Mobile)
  Supabase Auth - PostgreSQL - Supabase Storage
  Gallery - Library - Characters - Brands - Credits
  Generation (Replicate) - Storyboards
        |
        | REST API (/api/desktop/*)
        v
Director's Desktop (Electron)
  Palette Sync Layer          Local Engine
  - Auth (3 methods)          - GPU Generation
  - Gallery browser           - Local gallery (unlimited)
  - Library/Characters        - NLE Editor
  - Credits display           - Job Queue
  - Push results back         - Settings

  FastAPI Backend (localhost:8010)
  + /api/sync/* routes for Palette connection
```

Key principle: Desktop works fully offline/standalone. Palette connection is additive.

---

## Phase 1: Foundation (Auth + Sync Infrastructure)

### Authentication — Three Methods

1. **Browser login (default)** — Click "Sign In," Desktop opens browser to Palette's Supabase auth page. After login, token returns to Electron via deep link (`directorsdesktop://auth/callback`). Same UI as Palette (email/password + Google OAuth + sign up).

2. **QR code pairing** — Desktop generates short-lived pairing code + QR. Scan from Palette mobile app (already logged in). Desktop receives auth token via polling.

3. **API key** — Generate in Palette Settings > Developer > API Keys. Paste into Desktop settings.

**Token storage:** Electron `safeStorage` API. Backend receives via `Authorization` header on sync routes.

**Offline behavior:** Local generation, gallery, NLE all functional without auth. Cloud features appear only when connected.

### Sync Protocol

**New Palette API routes** (`/api/desktop/*`):

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/desktop/auth/validate` | POST | Validate Desktop auth token |
| `/api/desktop/gallery` | GET | Paginated gallery browse |
| `/api/desktop/gallery/download/:id` | GET | Download single asset |
| `/api/desktop/gallery/upload` | POST | Push local asset to cloud |
| `/api/desktop/library/characters` | GET | List characters + references |
| `/api/desktop/library/styles` | GET | List style guides + brands |
| `/api/desktop/library/references` | GET | List reference images |
| `/api/desktop/credits` | GET | Credit balance |
| `/api/desktop/prompts` | GET | Synced prompt library |
| `/api/desktop/send-job` | POST | Send generation job from Palette to Desktop |

**New Desktop API routes** (`/api/sync/*`):

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/sync/connect` | POST | Store Palette auth token |
| `/api/sync/status` | GET | Connection status + user info |
| `/api/sync/receive-job` | POST | Accept generation job from Palette |

---

## Phase 2: Gallery + Library

### Gallery — Local + Cloud Unified View

Two tabs:
- **Local** — Unlimited images/videos from local generations. Stored on disk.
- **Cloud** — Paginated browse of Palette gallery (500 image cap server-side). Pull on demand.

Key behaviors:
- Cloud items show cloud badge. Click to download locally.
- Local items have "Push to Cloud" button (counts against 500 cap).
- Delete local ≠ delete cloud. Independent.
- Search/filter across both tabs.
- Model badge on each asset (already implemented).

### Library — Characters, Styles, References

**Characters panel:**
- Pull character list from Palette (name, role, reference images).
- Create local-only characters (name + reference images).
- Push individual local characters to Palette when ready.
- Pick a character when generating → references auto-attach.

**Styles panel:**
- Browse Palette's style guides and brand visual identities.
- Create local style presets (reference image + description).
- Generate Style Guide Grid (3x3).

**References panel:**
- Categorized reference images (people, places, props).
- `@mention` autocomplete in prompt field.
- Pull from Palette or upload locally.

### Credits Display
- Live balance visible in sidebar.
- Deducted when using cloud generation (Replicate API).
- Local GPU generation is free (no credits needed).

---

## Phase 3: Generation Upgrades

### First Frame + Last Frame

Two image slots in Playground:

```
[First Frame]    [Last Frame]
Paste / Drop     Paste / Drop
or Browse        or Browse
```

**Input methods:** Ctrl+V paste from clipboard, drag & drop, file browse, or "Extract Frame" from existing video.

**Backend wiring:**
- LTX local: First → `ImageConditioningInput(frame_idx=0)`, Last → `ImageConditioningInput(frame_idx=num_frames-1)`. Both can be set simultaneously.
- Seedance (Replicate): First → `image` param, Last → `last_frame` param.

**New API fields:**
- `GenerateVideoRequest.lastFramePath: str | None = None`
- `QueueSubmitRequest` updated to pass last frame path through params.

### Image Variations
- Add "Variations: 1-12" slider to image generation settings.
- Backend already supports it (clamped 1-12), UI just needs to expose it.
- Results displayed in a grid.

### Social Media Aspect Ratio Presets
- 16:9 → "YouTube / Landscape"
- 9:16 → "TikTok / Reels / Shorts"
- 1:1 → "Instagram Square"
- 4:3 → "Standard"
- 4:5 → "Instagram Post" (new)

### Prompt Enhancement Button
- Sparkle button next to prompt field.
- Uses Gemini (key already in settings) to enhance rough prompt into detailed cinematic prompt.
- New backend route: `POST /api/enhance-prompt` (generalize existing gap-prompt logic).

### Frame Extraction from Player
- Right-click video in player → "Extract Frame" → saves to local gallery.
- "Use as First Frame" / "Use as Last Frame" context menu options.
- Uses existing `extract-video-frame` IPC handler.

---

## Phase 4: Power Tools

### Wildcards
- Define variation lists: `_outfit_` = ["red dress", "blue suit", "leather jacket"]
- Use in prompts: "A woman in _outfit_ walking through _location_"
- Generate all combinations or random selection.
- Parser logic from Palette's `wildcard/parser.ts`.

### Prompt Library with Autocomplete
- Save and recall prompts with `@tag` autocomplete.
- Sort by most-used, recent, category.
- Sync with Palette prompt library when connected.

### Contact Sheet Generation
- Select reference image → generate 3x3 grid of cinematic angles in one API call.
- Slice into 9 separate images.
- Add any to project timeline or gallery.

### Style Guide Grids
- Reference image + style name → 3x3 grid showing style across diverse subjects.
- Store in project asset library.

---

## Phase 5: Advanced Features

### Send-to-Desktop from Palette Web
- Button in Palette: "Send to Desktop" on any generation prompt or storyboard shot.
- Desktop receives via `/api/sync/receive-job`.
- Job appears in Desktop queue with prompt + references pre-loaded.

### Inpainting in Editor
- Annotation canvas overlay on video frames.
- Draw masks/arrows/labels → send to inpaint API.
- Non-destructive editing of generated shots.

### Bidirectional Prompt/Job Sync
- Generation jobs started in Palette can be monitored in Desktop and vice versa.
- Shared job history when connected.

### QR Code Pairing
- Full implementation with WebSocket handshake.
- One-time scan, persistent session.

---

## Phase 6: Testing & Quality Assurance

### Backend Integration Tests (Python — pytest)

**Auth & sync tests:**
- Token validation (valid, expired, malformed)
- Connection status (connected, disconnected, token refresh)
- Auth token storage and retrieval

**Gallery sync tests:**
- Cloud gallery pagination and filtering
- Asset download (success, not found, unauthorized)
- Asset upload/push (success, quota exceeded, duplicate)
- Local delete does not affect cloud
- Cloud browse while offline (graceful failure)

**Library sync tests:**
- Character list fetch and caching
- Character reference image download
- Style guide fetch
- Reference image categorization
- Push local character to cloud

**Generation pipeline tests:**
- First frame only generation (LTX + Seedance)
- Last frame only generation (LTX + Seedance)
- First + last frame simultaneous generation
- Last frame index calculation from num_frames
- Paste/clipboard image handling
- Image variations (1, 4, 12 count)
- Prompt enhancement route

**Queue tests:**
- Receive job from Palette (valid, malformed)
- Job with character references attached
- Job with first/last frame images
- Credits check before cloud generation

### Frontend Tests (TypeScript)

**Component tests:**
- First/Last frame image slots (paste, drop, browse, clear)
- Gallery tab switching (local vs cloud)
- Character picker with reference previews
- Credits display (connected vs disconnected)
- Sidebar navigation states (auth vs no-auth)
- Image variation grid rendering
- Social media preset labels
- Prompt autocomplete dropdown

**Integration tests:**
- Auth flow: browser login → token stored → sync routes work
- Gallery: browse cloud → download → appears in local
- Push: local asset → push to cloud → appears in cloud tab
- Generation: set first+last frame → generate → correct params sent

### End-to-End Tests

- Full auth → browse gallery → download → use as first frame → generate → push result workflow
- Offline mode: all local features work without Palette connection
- Reconnection: lose connection → queue local work → reconnect → sync

### Palette API Tests (Next.js — in Director's Palette repo)

- Desktop auth validation endpoint
- Gallery pagination with RLS (user sees only their data)
- Asset download with signed URLs
- Upload with quota enforcement
- Character/style/reference list endpoints
- Credits balance endpoint
- Send-job endpoint

---

## Sidebar Layout

```
[Logo] Director's Desktop

── CREATE ──────────
  Playground         (generate images/video)
  Queue              (job queue status)

── EDIT ────────────
  Projects           (NLE timeline projects)

── LIBRARY ─────────
  Gallery            (local + cloud)
  Characters         (Palette + local)
  Styles             (style guides/brands)
  References         (categorized refs)

── TOOLS ───────────
  Wildcards          (prompt variations)
  Prompt Library     (saved prompts)
  Contact Sheets     (3x3 angle grids)

── ACCOUNT ─────────
  Credits: 4,250     (live balance)
  Settings
  Sign In / User
```

When not signed in: LIBRARY shows local-only items + "Sign in to sync" CTA. ACCOUNT shows Sign In button. Credits hidden.

---

## Key Decisions

1. **Desktop works standalone** — No auth required for local generation/editing.
2. **Cloud is additive** — Palette connection unlocks gallery, library, credits, sync.
3. **Delete local ≠ delete cloud** — One-way protection.
4. **Push is explicit** — User chooses what to upload to cloud.
5. **Credits only for cloud generation** — Local GPU is free.
6. **LTX supports last frame** — via `frame_idx` parameter on `ImageConditioningInput`.
7. **Three auth methods** — Browser login (default), QR pairing (mobile), API key (power users).
