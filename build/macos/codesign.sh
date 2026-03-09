#!/usr/bin/env bash
# build/macos/codesign.sh
# Ad-hoc code signing for LTX Desktop macOS app bundle.
#
# No Apple Developer account or Apple ID is required. By default this script
# uses ad-hoc signing (identity "-"), which prevents the "app is damaged" Gatekeeper
# warning without needing a paid Apple Developer certificate.
#
# For production releases with a real Developer ID, set the MACOS_SIGNING_IDENTITY
# environment variable to your certificate name (e.g. "Developer ID Application: …").
#
# Adapted from: https://github.com/audiohacking/AceForge/blob/main/build/macos/codesign.sh
# Original reference: https://github.com/dylanwh/lilguy/blob/main/macos/build.sh
#
# Usage:
#   bash build/macos/codesign.sh <path-to-app.bundle>

set -euo pipefail

APP_PATH="${1:?Usage: $0 <path-to-app.bundle>}"
SIGNING_IDENTITY="${MACOS_SIGNING_IDENTITY:--}"   # Default: ad-hoc ("-")
ENTITLEMENTS_PATH="resources/entitlements.mac.plist"

echo "=================================================="
echo "  LTX Desktop macOS Code Signing"
echo "=================================================="
echo "  App:          $APP_PATH"
echo "  Identity:     $SIGNING_IDENTITY"
echo "  Entitlements: $ENTITLEMENTS_PATH"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

if [ ! -f "$ENTITLEMENTS_PATH" ]; then
    echo "Error: Entitlements file not found at $ENTITLEMENTS_PATH"
    echo "       Run this script from the project root."
    exit 1
fi

# ── Signing helper ────────────────────────────────────────────────────────────
# Ad-hoc signing ("-") does not support --timestamp (requires a CA).
sign_target() {
    local target="$1"
    echo "  Signing: $(basename "$target")"

    if [ "$SIGNING_IDENTITY" = "-" ]; then
        xcrun codesign \
            --sign "$SIGNING_IDENTITY" \
            --force \
            --options runtime \
            --entitlements "$ENTITLEMENTS_PATH" \
            --deep \
            "$target"
    else
        xcrun codesign \
            --sign "$SIGNING_IDENTITY" \
            --force \
            --options runtime \
            --entitlements "$ENTITLEMENTS_PATH" \
            --deep \
            --timestamp \
            "$target"
    fi

    echo "  ✓ Signed: $(basename "$target")"
}

# ── Step 1: Sign bundled Python native libraries ──────────────────────────────
# Sign leaf Mach-O binaries first so the bundle signature remains valid.
# The Python embed directory contains .dylib/.so native extensions.
echo "Step 1: Signing bundled Python native libraries..."
PYTHON_DIR="$APP_PATH/Contents/Resources/python"
if [ -d "$PYTHON_DIR" ]; then
    find "$PYTHON_DIR" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 | \
        while IFS= read -r -d '' lib; do
            sign_target "$lib" || true   # keep going on individual failures
        done
    echo "  Python native libraries signed."
else
    echo "  No bundled Python directory found at Contents/Resources/python — skipping."
fi

# ── Step 2: Sign Electron framework libraries ────────────────────────────────
echo ""
echo "Step 2: Signing Electron framework libraries..."
if [ -d "$APP_PATH/Contents/Frameworks" ]; then
    find "$APP_PATH/Contents/Frameworks" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 | \
        while IFS= read -r -d '' f; do
            sign_target "$f" || true
        done
fi

# ── Step 3: Sign main executables ────────────────────────────────────────────
echo ""
echo "Step 3: Signing main executables..."
for exe in "$APP_PATH/Contents/MacOS/"*; do
    if [ -f "$exe" ] && [ -x "$exe" ]; then
        sign_target "$exe"
    fi
done

# ── Step 4: Sign the whole app bundle ────────────────────────────────────────
echo ""
echo "Step 4: Signing app bundle..."
sign_target "$APP_PATH"

# ── Verification ─────────────────────────────────────────────────────────────
echo ""
echo "Verification:"
xcrun codesign --verify --deep --strict --verbose=2 "$APP_PATH" 2>&1 || true

echo ""
echo "Signature info:"
xcrun codesign -dv "$APP_PATH" 2>&1 || true

echo ""
echo "=================================================="
echo "  ✓ Code signing complete!"
echo "=================================================="
