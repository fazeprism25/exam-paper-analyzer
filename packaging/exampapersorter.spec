"""PyInstaller build spec for the desktop GUI (onedir mode).

Onedir (a folder of files) rather than onefile: onefile re-extracts the
whole bundle to a temp directory on every launch, which is slow and
wasteful for an app this size (docling pulls in torch + torchvision,
several hundred MB), and onedir plays more predictably with antivirus
scanners that are suspicious of self-extracting single-file executables.

Build with:
    pyinstaller packaging/exampapersorter.spec --noconfirm

Run from the repository root so relative paths below resolve correctly.

`collect_all` (rather than hand-picked hiddenimports) is used for every
package with dynamic/plugin-style imports or non-Python resource files
(model configs, tokenizer data, native shared libraries) that a plain
static import scan would miss -- reliability with Docling/FastEmbed/ONNX
Runtime is the top packaging priority here, not installer size.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

REPO_ROOT = Path(SPECPATH).resolve().parent
APP_NAME = "ExamPaperAnalyzer"

_COLLECT_ALL_PACKAGES = [
    "docling",
    "docling_core",
    "docling_ibm_models",
    "docling_parse",
    "rapidocr",
    "onnxruntime",
    "fastembed",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "transformers",
    "torch",
    "torchvision",
]

datas = [
    (str(REPO_ROOT / "config" / "openrouter_models.json"), "config"),
]
binaries = []
hiddenimports = [
    "keyring.backends",
    "keyring.backends.Windows",
    "keyring.backends.macOS",
    "keyring.backends.SecretService",
]

def _is_vendored_license_bloat(dest_str: str) -> bool:
    """torch (and to a lesser extent some of its neighbors) ships vendored
    third-party LICENSE *source* trees inside its own .dist-info metadata
    folder -- e.g. torch-*.dist-info/licenses/third_party/kineto/libkineto/
    third_party/dynolog/third_party/prometheus-cpp/3rdparty/googletest/
    googlemock/scripts/generator, many directories deep. Confirmed directly
    on this machine: PyInstaller's COLLECT step failed with WinError 206
    ("filename or extension is too long") trying to create that exact
    directory under dist/, and separately, Inno Setup's compiler aborted on
    the same deep paths with "The system cannot find the path specified."
    None of these files are ever imported or read by this app at runtime --
    only the flat top-level dist-info files (METADATA, RECORD) matter for the
    importlib.metadata lookups libraries like transformers make -- so the
    deep vendored subtree is safe to drop without losing anything the app
    actually needs.
    """
    if ".dist-info" in dest_str and ("license" in dest_str.lower() or "third_party" in dest_str.lower()):
        return True
    # Generic safety net against any other similarly-deep vendored path
    # (Windows MAX_PATH is 260 chars; leave headroom for the install dir).
    return len(dest_str) > 180


def _drop_vendored_license_bloat(pkg_datas):
    """Filters (source, dest) 2-tuples as returned by collect_all()."""
    return [(source, dest) for source, dest in pkg_datas if not _is_vendored_license_bloat(str(dest))]


def _drop_vendored_license_bloat_toc(toc_entries):
    """Filters the (dest, source, typecode) 3-tuples PyInstaller's Analysis()
    produces in a.datas -- a different shape than collect_all()'s 2-tuples,
    and populated separately (see comment at the a.datas assignment below).
    """
    return [(dest, source, typecode) for dest, source, typecode in toc_entries if not _is_vendored_license_bloat(str(dest))]


for package in _COLLECT_ALL_PACKAGES:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += _drop_vendored_license_bloat(pkg_datas)
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    [str(REPO_ROOT / "exampapersorter" / "app.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    # botocore/boto3/s3fs are an unrelated AWS SDK -- this app never talks to
    # AWS, only OpenRouter (plain `requests`) and local model files. They
    # get pulled in transitively (huggingface_hub/fsspec optionally support
    # S3-backed hub storage) and PyInstaller's botocore hook then bundles
    # its per-AWS-service JSON data files (hundreds of MB across every
    # region/service), which is exactly what triggered a Windows path-length
    # failure during COLLECT on this machine. Excluding them entirely is
    # safe: nothing in this app's own code imports boto3/botocore/s3fs.
    excludes=["matplotlib", "notebook", "jupyter", "boto3", "botocore", "s3fs", "s3transfer", "awscrt"],
    noarchive=False,
)

# PyInstaller auto-runs each dependency's own bundled hook during Analysis()
# above, independently of the collect_all() calls this spec makes -- torch's
# hook calls copy_metadata(..., recursive=True), which re-adds the same deep
# vendored license tree straight into a.datas, bypassing the
# _drop_vendored_license_bloat() filter applied to the collect_all() results
# earlier. Confirmed on GitHub Actions: PyInstaller's own COLLECT step
# tolerates the resulting long paths, but Inno Setup's compiler does not, and
# aborts with "The system cannot find the path specified." Filtering the
# final merged a.datas (not just the pre-Analysis datas list) catches both
# sources.
a.datas = _drop_vendored_license_bloat_toc(a.datas)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(REPO_ROOT / "packaging" / "icon" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if sys.platform == "darwin":
    icns_path = REPO_ROOT / "packaging" / "icon" / "icon.icns"
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=str(icns_path) if icns_path.exists() else None,
        bundle_identifier="com.exampapersorter.desktop",
        info_plist={
            "CFBundleName": "Exam Paper Analyzer",
            "CFBundleDisplayName": "Exam Paper Analyzer",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    )
