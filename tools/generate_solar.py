#!/usr/bin/env python3
"""Generate Solar's deterministic RGB565 poses from the supplied artwork."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

if __package__:
    from . import character_pack, sprite_codec
else:
    import character_pack
    import sprite_codec

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "characters" / "solar"
TRACKS = (
    "right", "left", "up", "down", "up_right", "up_left",
    "down_right", "down_left", "surprise", "working", "complete",
    "attention", "sleep",
)
LOOKS = ((1, 0), (-1, 0), (0, -1), (0, 1),
         (1, -1), (-1, -1), (1, 1), (-1, 1))
SIZE = (character_pack.WIDTH, character_pack.HEIGHT)
RESAMPLE = Image.Resampling.LANCZOS
SUBJECT_SIZE = 272
# Measured on the supplied 1024x1024 artwork.
DISC_CENTER, DISC_RADIUS = (512, 490), 231
GLOW_COLOR = (255, 186, 40)

# In solar mode, solar-agent maps each solar level to one of these tracks.
NEUTRAL = dict(scale=0.85, rays=0.68, light=0.86, saturation=0.72,
               warmth=0.10, glow=0.0, cloud=0.0, night=0.0, ring=0.0)
LEVELS = {
    8: dict(NEUTRAL, scale=0.96, rays=0.96, light=1.0, saturation=0.90),
    9: dict(NEUTRAL, scale=1.0, rays=1.08, light=1.08, saturation=1.10,
            glow=0.28, ring=0.35),
    10: dict(NEUTRAL, scale=1.20, rays=1.25, light=1.20, saturation=1.18,
             warmth=0.75, glow=1.0, ring=1.0),
    11: dict(NEUTRAL, scale=0.90, rays=0.52, light=0.40, saturation=0.18,
             cloud=1.0),
    12: dict(NEUTRAL, scale=0.85, rays=0.0, light=0.29, saturation=0.08,
             warmth=0.0, night=1.0),
}
SOLAR_LABELS = {
    "sleep": "night / no production",
    "attention": "importing in daylight",
    "idle": "producing, below minimum",
    "working": "minimum surplus",
    "complete": "max surplus",
    "surprise": "tap",
}
SOLAR_ORDER = ("sleep", "attention", "idle", "working", "complete")


def rgb565(image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint16)
    words = ((pixels[:, :, 0] >> 3) << 11
             | (pixels[:, :, 1] >> 2) << 5 | pixels[:, :, 2] >> 3)
    return words.astype(">u2").tobytes()


def compress(raw, width):
    words = np.frombuffer(raw, dtype=">u2").astype(np.uint16).reshape(-1, width)
    residual = words.copy()
    residual[1:] = words[1:] - words[:-1]
    return zlib.compress(residual.astype(">u2").tobytes(), 9)


def prepare_artwork():
    """Return the eyeless sun, its movable eye layers, and the disc center/radius."""
    source = Image.open(DIRECTORY / "source" / "solar.png").convert("RGB")
    if source.size != (1024, 1024):
        raise ValueError("Solar's eye and disc coordinates require the supplied 1024x1024 artwork.")
    pixels = np.asarray(source, dtype=np.int16)
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    alpha = Image.fromarray(
        np.clip((saturation.astype(np.float32) - 12) * (255 / 24), 0, 255).astype("uint8")
    )
    subject = source.convert("RGBA")
    subject.putalpha(alpha)
    bounds = alpha.getbbox()
    if bounds is None:
        raise ValueError("No colorful foreground found in the source image.")
    subject = subject.crop(bounds)
    # Leaves room for the max-surplus rays and glow inside the 412x352 canvas.
    scale = SUBJECT_SIZE / max(subject.size)
    subject = subject.resize(tuple(round(side * scale) for side in subject.size), RESAMPLE)
    offset = ((SIZE[0] - subject.width) // 2, (SIZE[1] - subject.height) // 2 - 14)
    original = Image.new("RGB", SIZE)
    original.paste(subject, offset, subject)

    def place(x, y):
        return (x - bounds[0]) * scale + offset[0], (y - bounds[1]) * scale + offset[1]

    eyes = []
    clean = original.copy()
    for left, top, right, bottom in ((430, 440, 488, 535), (535, 440, 594, 535)):
        box = tuple(round(value) for value in (*place(left, top), *place(right, bottom)))
        tile = original.crop(box)
        colors = np.asarray(tile)
        mask = Image.fromarray(
            ((colors[:, :, 1] > 115) & (colors[:, :, 2] > 175)).astype("uint8") * 255
        ).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1))
        # Interpolate dark visor pixels across the eye, leaving its outer glow intact.
        backdrop = Image.new("RGB", (2, tile.height))
        backdrop.paste(tile.crop((0, 0, 1, tile.height)), (0, 0))
        backdrop.paste(tile.crop((tile.width - 1, 0, tile.width, tile.height)), (1, 0))
        backdrop = backdrop.resize(tile.size, Image.Resampling.BILINEAR)
        clean.paste(backdrop, box, mask)
        eye = tile.convert("RGBA")
        eye.putalpha(mask)
        eyes.append((box, eye))
    center = place(*DISC_CENTER)
    return clean, eyes, (center[0], center[1], DISC_RADIUS * scale)


def level(direction, depth):
    """Interpolate a track's look from the shared neutral center pose."""
    target = LEVELS.get(direction, NEUTRAL)
    return {key: NEUTRAL[key] + (target[key] - NEUTRAL[key]) * depth for key in NEUTRAL}


def scale_around(image, center, radius, scale):
    if scale == 1.0:
        return image
    cx, cy = center
    yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    distance = np.hypot(dx, dy)
    # Enlarge the bright disc fully while tapering the warp across its rays.
    factor = 1 + (scale - 1) * np.exp(-((distance / (radius * 1.6)) ** 4))
    sx, sy = cx + dx / factor, cy + dy / factor
    valid = (sx >= 0) & (sx < SIZE[0] - 1) & (sy >= 0) & (sy < SIZE[1] - 1)
    x0, y0 = np.clip(sx.astype(np.int32), 0, SIZE[0] - 2), np.clip(sy.astype(np.int32), 0, SIZE[1] - 2)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    source = np.asarray(image, dtype=np.float32)
    top = source[y0, x0] * (1 - fx) + source[y0, x0 + 1] * fx
    bottom = source[y0 + 1, x0] * (1 - fx) + source[y0 + 1, x0 + 1] * fx
    pixels = (top * (1 - fy) + bottom * fy) * valid[..., None]
    return Image.fromarray(np.clip(pixels, 0, 255).round().astype("uint8"))


def stretch_rays(clean, sun, factor):
    """Lengthen or retract the rays radially while keeping the disc unchanged."""
    if factor == 1:
        return np.asarray(clean, dtype=np.float32)
    cx, cy, radius = sun
    source = np.asarray(clean, dtype=np.float32)
    yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    distance = np.hypot(dx, dy)
    sample = np.where(distance > radius, radius + (distance - radius) / factor, distance)
    ratio = sample / np.maximum(distance, 1e-6)
    sx = np.clip(cx + dx * ratio, 0, SIZE[0] - 1.001)
    sy = np.clip(cy + dy * ratio, 0, SIZE[1] - 1.001)
    x0, y0 = sx.astype(np.int32), sy.astype(np.int32)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    top = source[y0, x0] * (1 - fx) + source[y0, x0 + 1] * fx
    bottom = source[y0 + 1, x0] * (1 - fx) + source[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def body(clean, sun, direction, step, count, cache):
    """Return one complete, level-specific sun body shared by the blink frames."""
    key = (direction, step)
    if key not in cache:
        look = level(direction, step / (count - 1))
        cx, cy, radius = sun
        pixels = stretch_rays(clean, sun, max(look["rays"], 0.5))
        yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]].astype(np.float32)
        distance = np.hypot(xx - cx, yy - cy)
        if look["night"]:
            # Keep the central disc as a dark night-light and remove the sun rays.
            pixels[distance > radius * 1.06] = 0
            pixels[distance <= radius * 1.06] *= np.array([0.48, 0.56, 0.82], dtype=np.float32)
        silhouette = Image.fromarray((pixels.max(axis=2) > 8).astype("uint8") * 255)
        halo = np.asarray(silhouette.filter(ImageFilter.GaussianBlur(12)), dtype=np.float32) / 255
        cover = np.asarray(silhouette.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32) / 255
        glow = (np.array(GLOW_COLOR, dtype=np.float32) * (look["glow"] * 1.2)
                * halo[..., None] * (1 - cover[..., None]))
        pixels = np.asarray(Image.fromarray(
            np.clip(pixels + glow, 0, 255).round().astype("uint8")
        ), dtype=np.float32)
        gray = (pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        pixels = gray + (pixels - gray) * look["saturation"]
        pixels *= look["light"]
        pixels *= np.array([1, 1 + 0.05 * look["warmth"],
                            1 - 0.52 * look["warmth"]], dtype=np.float32)
        if direction == 10:
            core = np.exp(-((distance / (radius * 0.78)) ** 4))[..., None]
            pixels += core * np.array([48, 50, 8], dtype=np.float32) * look["glow"]
        rendered = Image.fromarray(np.clip(pixels, 0, 255).round().astype("uint8"))
        rendered = scale_around(rendered, (cx, cy), radius, look["scale"])
        cache[key] = (rendered, look)
    return cache[key]


def draw_cloud(image, sun, amount):
    """Drift a soft cloud over the lower-left rays for daytime grid import."""
    if amount <= 0:
        return image
    cx, cy, radius = sun
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x = cx - radius * (0.55 + 0.9 * (1 - amount))
    y = cy + radius * 0.82
    fill = (226, 231, 240, round(235 * amount))
    shade = (168, 177, 193, round(235 * amount))
    for ox, oy, size in ((-0.42, 0.06, 0.30), (0.0, -0.12, 0.40), (0.40, 0.04, 0.30)):
        r = radius * size
        draw.ellipse((x + ox * radius - r, y + oy * radius - r + 3,
                      x + ox * radius + r, y + oy * radius + r + 3), fill=shade)
    for ox, oy, size in ((-0.42, 0.06, 0.30), (0.0, -0.12, 0.40), (0.40, 0.04, 0.30)):
        r = radius * size
        draw.ellipse((x + ox * radius - r, y + oy * radius - r,
                      x + ox * radius + r, y + oy * radius + r), fill=fill)
    layer = layer.filter(ImageFilter.GaussianBlur(1.2))
    result = image.convert("RGBA")
    result.alpha_composite(layer)
    return result.convert("RGB")


def draw_solar_accents(image, sun, look, direction, step):
    """Add distinct night stars and surplus halos without obscuring the sun."""
    cx, cy, radius = sun
    scale = look["scale"]
    cx, cy, radius = cx, cy, radius * scale
    result = image.convert("RGBA")
    draw = ImageDraw.Draw(result)
    if look["night"] > 0:
        stars = ((0.12, 0.14, 2), (0.82, 0.13, 1), (0.91, 0.48, 2),
                 (0.11, 0.69, 1), (0.83, 0.82, 2), (0.31, 0.91, 1))
        for index, (sx, sy, size) in enumerate(stars):
            if (index + step) % 4 == 0:
                draw.line((sx * SIZE[0] - size, sy * SIZE[1],
                           sx * SIZE[0] + size, sy * SIZE[1]),
                          fill=(128, 178, 255, 210), width=1)
                draw.line((sx * SIZE[0], sy * SIZE[1] - size,
                           sx * SIZE[0], sy * SIZE[1] + size),
                          fill=(128, 178, 255, 210), width=1)
            else:
                draw.ellipse((sx * SIZE[0] - size, sy * SIZE[1] - size,
                              sx * SIZE[0] + size, sy * SIZE[1] + size),
                             fill=(150, 190, 255, 200))
    if look["ring"] > 0:
        angle = step * 360 / max(character_pack.MODEL_TRACK_STEPS[direction] - 1, 1)
        orbit = radius * 1.30
        bounds = (cx - orbit, cy - orbit, cx + orbit, cy + orbit)
        alpha = round(220 * look["ring"])
        draw.arc(bounds, angle, angle + 245,
                 fill=(255, 214, 89, alpha), width=max(1, round(2 * scale)))
        if direction == 10:
            outer = radius * 1.52
            draw.arc((cx - outer, cy - outer, cx + outer, cy + outer),
                     angle + 100, angle + 290,
                     fill=(255, 246, 191, alpha), width=max(1, round(2 * scale)))
            for index in range(8):
                theta = math.radians(angle + index * 45)
                sx, sy = cx + math.cos(theta) * outer, cy + math.sin(theta) * outer
                size = 1 + (index % 3 == step % 3)
                draw.line((sx - size, sy, sx + size, sy), fill=(255, 250, 205, alpha), width=1)
                draw.line((sx, sy - size, sx, sy + size), fill=(255, 250, 205, alpha), width=1)
    return result.convert("RGB")


def render(clean, eyes, sun, direction, step, blink, count, cache=None):
    cache = {} if cache is None else cache
    depth = step / (count - 1)
    dx = dy = 0
    expression_height = 1.0
    if direction < 8:
        x, y = LOOKS[direction]
        dx, dy = round(x * 6 * depth), round(y * 4 * depth)
    elif direction == 8:
        expression_height = 1 + .18 * depth
        dy = -round(2 * depth)
    elif direction == 9:
        expression_height = 1 - .12 * depth
        dy = -round(2 * depth)
    elif direction == 10:
        expression_height = 1 - .65 * depth
        dy = -round(3 * depth)
    elif direction == 11:
        expression_height = 1 - .2 * depth
        dx, dy = -round(4 * depth), round(3 * depth)
    elif direction == 12:
        expression_height = 1 - .92 * depth
        dy = round(3 * depth)
    sun_body, look = body(clean, sun, direction, step, count, cache)
    result = sun_body.copy()
    cx, cy, _ = sun
    cx, cy = round(cx), round(cy)
    scale = look["scale"]
    for box, eye in eyes:
        left = round(cx + (box[0] - cx) * scale + dx * scale)
        top = round(cy + (box[1] - cy) * scale + dy * scale)
        width = max(1, round(eye.width * scale))
        original_height = round(eye.height * scale)
        height = max(2, round(original_height * expression_height * (1 - blink * .23)))
        lid = eye.resize((width, height), RESAMPLE)
        result.paste(lid, (left, top + (original_height - height) // 2), lid)
    depth = step / (count - 1)
    if direction in (10, 12) and depth:
        accent = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(accent)
        alpha = round(255 * depth)
        eye_y = round(cy + (450 - 490) * SIZE[0] / 1024 * scale + dy * scale)
        if direction == 10:
            eye_y = round(cy + (488 - 490) * SIZE[0] / 1024 * scale)
            color = (255, 250, 193, alpha)
            for eye_x in (round(cx - 15 * scale), round(cx + 15 * scale)):
                draw.arc((eye_x - 5, eye_y - 4, eye_x + 5, eye_y + 5),
                         200, 340, fill=color, width=max(1, round(2 * scale)))
        else:
            color = (80, 156, 222, alpha)
            for eye_x in (round(cx - 15 * scale), round(cx + 15 * scale)):
                draw.line((eye_x - 4, eye_y, eye_x + 4, eye_y),
                          fill=color, width=max(1, round(2 * scale)))
        result = Image.alpha_composite(result.convert("RGBA"), accent).convert("RGB")
    result = draw_solar_accents(result, sun, look, direction, step)
    result = draw_cloud(result, sun, look["cloud"])
    return result


def generate():
    clean, eyes, sun = prepare_artwork()
    cache = {}
    # Each look changes monotonically, so track endpoints bound every pose.
    extremes = [render(clean, eyes, sun, 0, 0, 0, 24, cache)] + [
        render(clean, eyes, sun, direction, count - 1, 0, count, cache)
        for direction, count in enumerate(character_pack.MODEL_TRACK_STEPS)
    ]
    boxes = [image.getbbox() for image in extremes]
    x0 = max(0, min(box[0] for box in boxes) - 2)
    y0 = max(0, min(box[1] for box in boxes) - 2)
    x1 = min(SIZE[0], max(box[2] for box in boxes) + 2)
    y1 = min(SIZE[1], max(box[3] for box in boxes) + 2)
    base_bounds = (x0, y0, x1, y1)
    base_size = (x0, y0, x1 - x0, y1 - y0)
    # A fixed eye-only patch keeps all blink buffers below the legacy 7,452-pixel limit.
    patch = (min(box[0] for box, _ in eyes) - 7,
             min(box[1] for box, _ in eyes) - 9,
             max(box[2] for box, _ in eyes) + 7,
             max(box[3] for box, _ in eyes) + 9)
    px, py, pr, pb = patch
    if (pr - px) * (pb - py) > 7452:
        raise ValueError("Eye patch exceeds the legacy firmware RAM budget.")
    data = bytearray()
    blocks = {}

    def block(image):
        raw = rgb565(image)
        key = (image.width, raw)
        if key not in blocks:
            encoded = compress(raw, image.width)
            if sprite_codec.decode_pixels(encoded, image.width) != raw:
                raise ValueError("RGB565 compression round-trip failed.")
            blocks[key] = {"offset": len(data), "size": len(encoded)}
            data.extend(encoded)
        return dict(blocks[key])

    frames, preview_rows = [], []
    preview = DIRECTORY / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    for direction, (name, count) in enumerate(zip(TRACKS, character_pack.MODEL_TRACK_STEPS)):
        cache = {}
        for step in range(count):
            poses = [render(clean, eyes, sun, direction, step, blink, count, cache)
                     for blink in range(5)]
            frames.append({
                "base": block(poses[0].crop(base_bounds)),
                "patchX": px, "patchY": py, "patchWidth": pr - px, "patchHeight": pb - py,
                "blinks": [block(pose.crop(patch)) for pose in poses[1:]],
            })
            if step == count - 1:
                poses[0].save(preview / f"{name}.png")
                preview_rows.append((name, poses[0]))
    idle = render(clean, eyes, sun, 0, 0, 0, 24)
    idle.save(preview / "idle.png")
    preview_rows.insert(0, ("idle", idle))

    sheet = Image.new("RGB", (206 * 4, 198 * math.ceil(len(preview_rows) / 4)), "#101426")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(preview_rows):
        x, y = index % 4 * 206, index // 4 * 198
        sheet.paste(image.resize((206, 176), RESAMPLE), (x, y))
        label = f"{name}: {SOLAR_LABELS[name]}" if name in SOLAR_LABELS else name
        draw.text((x + 8, y + 178), label, fill="white")
    sheet.save(preview / "contact-sheet.png")

    levels = Image.new("RGB", (206 * len(SOLAR_ORDER), 198), "#101426")
    draw = ImageDraw.Draw(levels)
    for index, name in enumerate(SOLAR_ORDER):
        image = dict(preview_rows)[name]
        levels.paste(image.resize((206, 176), RESAMPLE), (index * 206, 0))
        draw.text((index * 206 + 8, 178), SOLAR_LABELS[name], fill="white")
    levels.save(preview / "solar-levels.png")

    animation = []
    for name in SOLAR_ORDER:
        direction = 0 if name == "idle" else TRACKS.index(name)
        count = character_pack.MODEL_TRACK_STEPS[direction]
        cache = {}
        steps = [0] * 12 if name == "idle" else list(range(count)) + [count - 1] * 14
        steps += [] if name == "idle" else list(range(count - 2, -1, -1))
        animation += [render(clean, eyes, sun, direction, step, 0, count, cache)
                      .resize((206, 176), RESAMPLE) for step in steps]
    animation[0].save(preview / "solar.gif", save_all=True, append_images=animation[1:],
                      duration=60, loop=0, disposal=2)

    metadata = {
        "width": SIZE[0], "height": SIZE[1], "baseBounds": list(base_size),
        "maxPatchPixels": (pr - px) * (pb - py),
        "trackSteps": character_pack.MODEL_TRACK_STEPS,
        "tracks": list(TRACKS), "solarLevels": SOLAR_LABELS, "frames": frames,
        "sourceSha256": hashlib.sha256((DIRECTORY / "source" / "solar.png").read_bytes()).hexdigest(),
    }
    character_pack.write_if_changed(DIRECTORY / "frames.bin", bytes(data))
    character_pack.write_if_changed(
        DIRECTORY / "frames.json", (json.dumps(metadata, indent=2) + "\n").encode()
    )
    character_pack.build(ROOT)


if __name__ == "__main__":
    try:
        generate()
    except (OSError, ValueError, KeyError) as error:
        print(f"Solar generation failed: {error}", file=sys.stderr)
        sys.exit(1)
