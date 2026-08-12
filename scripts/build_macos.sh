#!/usr/bin/env bash
# Builds the macOS desktop app (PyInstaller onedir + .app bundle) and
# ExamPaperAnalyzer.dmg. Must run on macOS -- PyInstaller does not
# cross-compile, and the .icns/hdiutil steps are macOS-only tools.
#
# Reproduces locally what .github/workflows/build-macos.yml runs on CI.
# Run from the repository root: ./scripts/build_macos.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== Installing build dependency (pyinstaller) =="
python3 -m pip install --upgrade pyinstaller

echo "== Building macOS iconset -> icon.icns =="
ICONSET_DIR="$REPO_ROOT/packaging/icon/iconset/icon.iconset"
ICNS_PATH="$REPO_ROOT/packaging/icon/icon.icns"
if [ -d "$ICONSET_DIR" ]; then
    iconutil -c icns "$ICONSET_DIR" -o "$ICNS_PATH"
else
    echo "WARNING: $ICONSET_DIR not found (run scripts/build_icon.py first) -- building without a custom icon."
fi

echo "== Cleaning previous build output =="
rm -rf "$REPO_ROOT/build" "$REPO_ROOT/dist"

echo "== Running PyInstaller =="
pyinstaller "$REPO_ROOT/packaging/exampapersorter.spec" --noconfirm --distpath "$REPO_ROOT/dist" --workpath "$REPO_ROOT/build"

APP_PATH="$REPO_ROOT/dist/ExamPaperAnalyzer.app"
if [ ! -d "$APP_PATH" ]; then
    echo "Expected build output not found: $APP_PATH" >&2
    exit 1
fi
echo "Built: $APP_PATH"

echo "== Building DMG =="
DMG_STAGING="$REPO_ROOT/dist/dmg_staging"
DMG_PATH="$REPO_ROOT/dist/ExamPaperAnalyzer.dmg"
rm -rf "$DMG_STAGING" "$DMG_PATH"
mkdir -p "$DMG_STAGING"
cp -R "$APP_PATH" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

hdiutil create -volname "Exam Paper Analyzer" \
    -srcfolder "$DMG_STAGING" \
    -ov -format UDZO \
    "$DMG_PATH"

echo "Done. DMG output: $DMG_PATH"
echo
echo "NOTE: this .app/.dmg is unsigned and not notarized (no Apple Developer"
echo "account configured for this project -- see README's macOS Gatekeeper"
echo "section). Gatekeeper will block a plain double-click on first launch;"
echo "the documented workaround is right-click -> Open."
