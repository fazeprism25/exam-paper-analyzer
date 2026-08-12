"""Generates the application icon from scratch with Pillow (already a
transitive dependency via docling) -- no external asset, no design tool.

Produces:
  packaging/icon/icon_1024.png         master image
  packaging/icon/icon.ico              Windows (multi-size)
  packaging/icon/iconset/icon_*.png    macOS iconset source (packaging/
                                        build_macos.sh turns this into
                                        icon.icns with iconutil, macOS-only)

Run manually after changing the design: `python scripts/build_icon.py`.
Not part of the normal test/build path.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ICON_DIR = _REPO_ROOT / "packaging" / "icon"

_BACKGROUND = (67, 56, 202)       # indigo
_BACKGROUND_DARK = (49, 40, 158)  # gradient shade
_PAPER = (250, 250, 252)
_LINE = (148, 152, 200)
_ACCENT = (34, 197, 94)           # green checkmark badge


def _draw_master(size: int = 1024) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = int(size * 0.06)
    radius = int(size * 0.22)
    for y in range(margin, size - margin):
        t = (y - margin) / (size - 2 * margin)
        r = int(_BACKGROUND[0] + (_BACKGROUND_DARK[0] - _BACKGROUND[0]) * t)
        g = int(_BACKGROUND[1] + (_BACKGROUND_DARK[1] - _BACKGROUND[1]) * t)
        b = int(_BACKGROUND[2] + (_BACKGROUND_DARK[2] - _BACKGROUND[2]) * t)
        draw.line([(margin, y), (size - margin, y)], fill=(r, g, b, 255))

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [margin, margin, size - margin, size - margin], radius=radius, fill=255
    )
    bg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bg.paste(img, (0, 0), mask)
    img = bg
    draw = ImageDraw.Draw(img)

    paper_w, paper_h = int(size * 0.42), int(size * 0.54)
    paper_x = (size - paper_w) // 2 - int(size * 0.03)
    paper_y = (size - paper_h) // 2
    fold = int(paper_w * 0.28)

    draw.polygon(
        [
            (paper_x, paper_y),
            (paper_x + paper_w - fold, paper_y),
            (paper_x + paper_w, paper_y + fold),
            (paper_x + paper_w, paper_y + paper_h),
            (paper_x, paper_y + paper_h),
        ],
        fill=_PAPER,
    )
    draw.polygon(
        [
            (paper_x + paper_w - fold, paper_y),
            (paper_x + paper_w, paper_y + fold),
            (paper_x + paper_w - fold, paper_y + fold),
        ],
        fill=(219, 220, 235),
    )

    line_x0 = paper_x + int(paper_w * 0.16)
    line_x1 = paper_x + paper_w - int(paper_w * 0.16)
    line_h = int(size * 0.018)
    for i in range(5):
        y0 = paper_y + int(paper_h * 0.32) + i * int(paper_h * 0.115)
        width = line_x1 if i % 3 != 2 else line_x0 + int((line_x1 - line_x0) * 0.55)
        draw.rounded_rectangle([line_x0, y0, width, y0 + line_h], radius=line_h // 2, fill=_LINE)

    badge_r = int(size * 0.135)
    badge_cx = paper_x + paper_w - int(badge_r * 0.15)
    badge_cy = paper_y + paper_h - int(badge_r * 0.15)
    draw.ellipse(
        [badge_cx - badge_r, badge_cy - badge_r, badge_cx + badge_r, badge_cy + badge_r],
        fill=_ACCENT, outline=_BACKGROUND_DARK, width=int(size * 0.01),
    )
    check_w = int(size * 0.018)
    draw.line(
        [
            (badge_cx - badge_r * 0.45, badge_cy),
            (badge_cx - badge_r * 0.12, badge_cy + badge_r * 0.4),
            (badge_cx + badge_r * 0.5, badge_cy - badge_r * 0.4),
        ],
        fill=_PAPER, width=check_w, joint="curve",
    )

    return img


def main() -> None:
    _ICON_DIR.mkdir(parents=True, exist_ok=True)
    master = _draw_master(1024)
    master.save(_ICON_DIR / "icon_1024.png")

    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    master.save(
        _ICON_DIR / "icon.ico",
        sizes=[(s, s) for s in ico_sizes],
    )

    iconset_dir = _ICON_DIR / "iconset" / "icon.iconset"
    iconset_dir.mkdir(parents=True, exist_ok=True)
    # Apple's required iconset naming convention (see `man iconutil`).
    mac_sizes = {
        "icon_16x16.png": 16, "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128, "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256, "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512, "icon_512x512@2x.png": 1024,
    }
    for name, px in mac_sizes.items():
        master.resize((px, px), Image.LANCZOS).save(iconset_dir / name)

    print(f"Wrote {_ICON_DIR / 'icon_1024.png'}")
    print(f"Wrote {_ICON_DIR / 'icon.ico'} ({len(ico_sizes)} sizes)")
    print(f"Wrote {iconset_dir} ({len(mac_sizes)} PNGs, for `iconutil -c icns` on macOS)")


if __name__ == "__main__":
    main()
