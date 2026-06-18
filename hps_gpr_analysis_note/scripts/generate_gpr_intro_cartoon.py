#!/usr/bin/env python3
"""Generate the Section 1.3 GPR workflow cartoon.

The figure is intentionally a schematic, not a sampled HPS spectrum.  It uses
smooth fake data to show the per-test-mass training exclusion and the local GP
prediction without implying that only narrow neighboring regions are used.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "methodology_figs" / "gpr_intro_cartoon.png"

W, H = 1800, 1040
PLOT = (170, 230, 1160, 760)
XMIN, XMAX = 20.0, 170.0
YMIN, YMAX = 0.0, 1.08
M_TEST = 90.0
WIN_HALF_WIDTH = 11.5


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_SUB = font(18)
F_AXIS = font(18)
F_AXIS_BOLD = font(19, True)
F_LABEL = font(20, True)
F_SMALL = font(17)
F_LEGEND = font(19)
F_LEGEND_BOLD = font(22, True)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    tw, th = text_size(draw, text, fnt)
    draw.text((xy[0] - tw / 2, xy[1] - th / 2), text, font=fnt, fill=fill)


def xmap(x: float) -> float:
    x0, _, x1, _ = PLOT
    return x0 + (x - XMIN) / (XMAX - XMIN) * (x1 - x0)


def ymap(y: float) -> float:
    _, y0, _, y1 = PLOT
    return y1 - (y - YMIN) / (YMAX - YMIN) * (y1 - y0)


def continuum(x: float) -> float:
    base = 0.16 + 0.82 * math.exp(-(x - XMIN) / 77.0)
    broad_shape = 0.045 * math.exp(-((x - 67.0) / 33.0) ** 2)
    shoulder = 0.035 * math.exp(-((x - 132.0) / 27.0) ** 2)
    return base + broad_shape + shoulder


def jitter(i: int) -> float:
    return 0.018 * math.sin(1.73 * i + 0.4) + 0.010 * math.sin(0.47 * i + 1.2)


def line_points(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    return [(xmap(x), ymap(y)) for x, y in zip(xs, ys)]


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int],
    width: int = 4,
    head: int = 15,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - head * math.cos(angle - math.pi / 6),
        end[1] - head * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head * math.cos(angle + math.pi / 6),
        end[1] - head * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=fill)


def draw_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    angle: int = 90,
) -> None:
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    d = ImageDraw.Draw(dummy)
    box = d.textbbox((0, 0), text, font=fnt)
    tile = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8), (255, 255, 255, 0))
    td = ImageDraw.Draw(tile)
    td.text((4, 4), text, font=fnt, fill=fill)
    rotated = tile.rotate(angle, expand=True)
    image.alpha_composite(rotated, (xy[0] - rotated.width // 2, xy[1] - rotated.height // 2))


def main() -> None:
    bg = (250, 252, 254)
    ink = (39, 48, 61)
    muted = (98, 111, 130)
    grid = (225, 231, 238)
    frame = (148, 160, 176)
    blue = (43, 105, 176)
    amber = (247, 205, 92)
    amber_line = (172, 117, 33)
    green = (29, 130, 82)
    green_fill = (186, 224, 201)
    red = (209, 67, 67)

    image = Image.new("RGBA", (W, H), bg + (255,))
    draw = ImageDraw.Draw(image)

    # Header.
    draw.rounded_rectangle((130, 60, W - 130, 150), radius=14, fill=(246, 249, 252), outline=(205, 213, 224), width=2)
    draw.text((155, 84), "GPR background prediction at one test mass", font=F_TITLE, fill=ink)
    draw.text(
        (156, 126),
        "Train on smooth continuum outside the moving excluded window; predict the local background covariance.",
        font=F_SUB,
        fill=muted,
    )

    # Plot frame and grid.
    x0, y0, x1, y1 = PLOT
    draw.rectangle(PLOT, fill=(253, 254, 255), outline=frame, width=2)
    for x in [20, 50, 80, 110, 140, 170]:
        px = xmap(x)
        draw.line([(px, y0), (px, y1)], fill=grid, width=1)
        center_text(draw, (px, y1 + 26), str(x), F_SMALL, muted)
    for y, label in [(0.18, "low"), (0.50, "mid"), (0.82, "high")]:
        py = ymap(y)
        draw.line([(x0, py), (x1, py)], fill=grid, width=1)
        draw.text((x0 - 48, py - 10), label, font=F_SMALL, fill=muted)

    # Excluded test window.
    x_left = M_TEST - WIN_HALF_WIDTH
    x_right = M_TEST + WIN_HALF_WIDTH
    px_left = xmap(x_left)
    px_right = xmap(x_right)
    draw.rectangle((px_left, y0, px_right, y1), fill=amber + (105,), outline=amber_line + (255,), width=2)
    draw.line([(xmap(M_TEST), y0), (xmap(M_TEST), y1)], fill=(233, 180, 71), width=2)
    center_text(draw, (xmap(M_TEST), y1 + 42), "m_test", F_SMALL, amber_line)

    # Smooth reference curve, shown faintly so the fake scatter reads as a continuum.
    xs = [XMIN + i * (XMAX - XMIN) / 240 for i in range(241)]
    ys = [continuum(x) for x in xs]
    draw.line(line_points(xs, ys), fill=(190, 198, 207), width=5)
    draw.line(line_points(xs, ys), fill=(245, 247, 249), width=2)

    # Training scatter points outside the excluded window.
    train_xs = [XMIN + i * (XMAX - XMIN) / 64 for i in range(65)]
    for i, x in enumerate(train_xs):
        if x_left <= x <= x_right:
            continue
        y = continuum(x) + jitter(i)
        px, py = xmap(x), ymap(y)
        draw.ellipse((px - 4.2, py - 4.2, px + 4.2, py + 4.2), fill=(40, 46, 55))

    # GP prediction inside the excluded window.
    local_xs = [x_left + i * (x_right - x_left) / 80 for i in range(81)]
    local_mean = [continuum(x) - 0.045 * (x - x_left) / (x_right - x_left) for x in local_xs]
    local_sigma = [0.045 + 0.020 * math.exp(-((x - M_TEST) / 6.5) ** 2) for x in local_xs]
    upper = [m + s for m, s in zip(local_mean, local_sigma)]
    lower = [m - s for m, s in zip(local_mean, local_sigma)]
    band = line_points(local_xs, upper) + list(reversed(line_points(local_xs, lower)))
    draw.polygon(band, fill=green_fill + (190,), outline=green + (255,))
    draw.line(line_points(local_xs, local_mean), fill=green, width=5)

    # Illustrative narrow signal, only inside the excluded window.
    sig_y = [m + 0.18 * math.exp(-((x - M_TEST) / 3.1) ** 2) for x, m in zip(local_xs, local_mean)]
    draw.line(line_points(local_xs, sig_y), fill=red, width=4)

    # Training-region labels.  Keep all text outside the plot and use arrows into the two used regions.
    center_text(draw, (xmap(47), y0 - 50), "training region", F_LABEL, blue)
    arrow(draw, (xmap(47), y0 - 28), (xmap(58), y0 + 18), blue, width=4)
    center_text(draw, (xmap(134), y0 - 50), "training region", F_LABEL, blue)
    arrow(draw, (xmap(134), y0 - 28), (xmap(126), y0 + 18), blue, width=4)

    center_text(draw, (xmap(M_TEST), y0 - 50), "excluded from GP training", F_LABEL, amber_line)
    arrow(draw, (xmap(M_TEST), y0 - 28), (xmap(M_TEST), y0 + 50), amber_line, width=4)

    # Axis labels.
    center_text(draw, ((x0 + x1) / 2, y1 + 66), "invariant mass m", F_AXIS_BOLD, ink)
    draw_rotated_text(image, (x0 - 74, (y0 + y1) // 2), "events / bin", F_AXIS_BOLD, ink, angle=90)

    # Legend.
    leg = (1215, 300, 1670, 700)
    draw.rounded_rectangle(leg, radius=12, fill=(253, 254, 255), outline=(177, 187, 202), width=2)
    draw.text((1245, 328), "Legend", font=F_LEGEND_BOLD, fill=ink)
    ly = 382
    entries = [
        ("dot", "training data"),
        ("window", "excluded window"),
        ("band", "GP covariance band"),
        ("mean", "GP posterior mean"),
        ("signal", "illustrative narrow signal"),
    ]
    for kind, label in entries:
        if kind == "dot":
            draw.ellipse((1249, ly - 6, 1261, ly + 6), fill=(40, 46, 55))
        elif kind == "window":
            draw.rectangle((1247, ly - 12, 1265, ly + 12), fill=amber + (120,), outline=amber_line)
        elif kind == "band":
            draw.rectangle((1245, ly - 13, 1268, ly + 13), fill=green_fill + (210,), outline=green)
        elif kind == "mean":
            draw.line([(1244, ly), (1269, ly)], fill=green, width=5)
        elif kind == "signal":
            draw.line([(1244, ly), (1269, ly)], fill=red, width=4)
        draw.text((1292, ly - 12), label, font=F_LEGEND, fill=ink)
        ly += 60

    draw.text(
        (170, 850),
        "Cartoon, not data: the production fit uses all bins outside the moving excluded window,",
        font=F_SMALL,
        fill=muted,
    )
    draw.text(
        (170, 878),
        "with log-space preprocessing and resolution-scaled kernel bounds carried into the local likelihood.",
        font=F_SMALL,
        fill=muted,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
