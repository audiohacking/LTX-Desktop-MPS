# LTX Desktop - Installer Build Guide

This guide explains how to build a distributable installer for **LTX Desktop**.

- For running from source and debugging: see [`README.md`](../README.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
- For end-user requirements and first-run behavior: see [`README.md`](../README.md).

## What Gets Bundled

The installer includes:
- **Electron app** (React frontend + Electron shell)
- **Embedded Python** (version from [`backend/.python-version`](../backend/.python-version)) with all dependencies pre-installed:
  - PyTorch (CUDA on Windows, MPS on macOS)
  - FastAPI, Diffusers, Transformers
  - LTX-2 inference packages
  - All other required libraries
- **Backend Python code**

**NOT bundled** (downloaded at runtime):
- Model weights (downloaded on first run; can be large) from Hugging Face

The embedded Python is **fully isolated** from the target system's Python — it lives inside `{install_dir}/resources/python/` and never modifies system settings.

## Prerequisites

Before building, ensure you have:

1. **Node.js 18+** - https://nodejs.org/
2. **uv** - https://docs.astral.sh/uv/ (Python package manager)
3. **git** - needed for git-based Python packages
4. **Internet connection** (for downloading Python and packages)
5. **~15GB free space** (for Python environment + build artifacts)

### Platform-Specific

- **Windows**: PowerShell 5.1+ (comes with Windows 10/11)
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)

## Quick Build

### macOS
```bash
pnpm build:mac
```

### Windows
```powershell
pnpm build:win
```

This will:
1. Download a standalone Python distribution (version from [`backend/.python-version`](../backend/.python-version))
2. Install all Python dependencies (~10GB on Windows with CUDA, ~2-3GB on macOS with MPS)
3. Build the frontend
4. Package everything with electron-builder
5. Create a DMG (macOS) or NSIS installer (Windows) in the `release/` folder

## Build Options

### macOS

```bash
# Full build
pnpm build:mac

# Skip Python setup (if already prepared)
pnpm build:mac:skip-python

# Fast rebuild (unpacked, skip Python + pnpm install)
pnpm build:fast:mac

# Just prepare Python environment
pnpm prepare:python:mac
```

### Windows

```powershell
# Full build
pnpm build:win

# Skip Python setup (if already prepared)
pnpm build:win:skip-python

# Just prepare Python environment
pnpm prepare:python:win

# Fast rebuild (unpacked, skip Python + pnpm install)
pnpm build:fast:win

# Clean build
powershell -File scripts/local-build.ps1 -Clean
```

### Build Script Options

The `local-build.sh` script accepts:
- `--platform mac|win` — Target platform (auto-detected if omitted)
- `--skip-python` — Use existing `python-embed/` directory
- `--clean` — Remove build artifacts before starting
- `--unpack` — Build unpacked app only (faster, no installer/DMG)

## Build Output

### macOS
```
release/
  └── LTX Desktop-<version>-arm64.dmg
```

### Windows
```
release/
  └── LTX Desktop-<version>-Setup.exe
```

## Application Icon

Place icon files in `resources/` before building:
- `icon.ico` — Windows (multi-size ICO: 256x256, 128x128, 64x64, 48x48, 32x32, 16x16)
- `icon.png` — macOS (1024x1024 recommended)

## Troubleshooting

### "Python not found" during build
Ensure you have internet access. The script downloads Python automatically.

### Build fails with CUDA errors
The build doesn't require a GPU. CUDA packages are pre-built binaries.

### macOS: Gatekeeper "unidentified developer" warning
The distributed DMG is ad-hoc signed (no Apple Developer account), so Gatekeeper will
block the first launch. No shell commands are required:

1. Open the DMG and drag **LTX Desktop** to `/Applications`.
2. Double-click the app — macOS will show "cannot be opened because the developer cannot be verified."  Click **Cancel** (or **OK**) to dismiss.
3. Open **System Settings → Privacy & Security**. Scroll down to find the blocked app and click **Open Anyway**.
4. Confirm by clicking **Open** in the follow-up dialog.

The app will launch normally from this point on.

### Installer is too large
Expected sizes:
- **Windows**: ~10GB (PyTorch CUDA ~2.5GB + ML libraries ~5GB + Python ~200MB + Electron ~100MB)
- **macOS**: ~2-3GB (PyTorch MPS is much smaller than CUDA variant)

### Runtime / first-run issues
End-user topics like system requirements, first-run setup, and model download behavior are documented in [`README.md`](../README.md).

## Advanced: Manual Build Steps

### macOS
```bash
# 1. Prepare Python environment
bash scripts/prepare-python.sh

# 2. Install dependencies
pnpm install

# 3. Build frontend
pnpm build:frontend

# 4. Build DMG
npx electron-builder --mac

# Or build unpacked app (faster, for testing)
npx electron-builder --mac --dir
```

### Windows
```powershell
# 1. Prepare Python environment
./scripts/prepare-python.ps1

# 2. Install dependencies
pnpm install

# 3. Build frontend
pnpm build:frontend

# 4. Build installer
npx electron-builder --win
```
