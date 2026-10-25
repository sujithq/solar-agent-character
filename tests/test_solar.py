import json
import unittest

from tools import character_pack, generate_solar, sprite_codec


class SolarPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = generate_solar.DIRECTORY
        cls.metadata = json.loads((cls.folder / "frames.json").read_text())
        cls.data = (cls.folder / "frames.bin").read_bytes()

    def test_pack_matches_sources_and_partition(self):
        pack = character_pack.build_character(generate_solar.ROOT, self.folder)
        self.assertEqual(
            pack, (generate_solar.ROOT / "build" / "characters" / "solar.acpk").read_bytes()
        )
        info = character_pack.decode(
            pack, character_pack.read_assets_partition(generate_solar.ROOT)["size"]
        )
        self.assertEqual(info["id"], "solar")
        self.assertEqual(info["frame_count"], 288)
        self.assertGreater(info["thumbnail_bytes"], 0)

    def test_every_frame_decodes_and_patches_are_contained(self):
        metadata = self.metadata
        bx, by, bw, bh = metadata["baseBounds"]
        self.assertLessEqual(metadata["maxPatchPixels"], 7452)
        self.assertEqual(len(metadata["frames"]), sum(character_pack.MODEL_TRACK_STEPS))
        for frame in metadata["frames"]:
            base = frame["base"]
            decoded = sprite_codec.decode_pixels(
                self.data[base["offset"]:base["offset"] + base["size"]], bw
            )
            self.assertEqual(len(decoded), bw * bh * 2)
            px, py, pw, ph = (frame[key] for key in (
                "patchX", "patchY", "patchWidth", "patchHeight"
            ))
            self.assertTrue(bx <= px < px + pw <= bx + bw)
            self.assertTrue(by <= py < py + ph <= by + bh)
            self.assertEqual(pw * ph, metadata["maxPatchPixels"])
            for block in frame["blinks"]:
                decoded = sprite_codec.decode_pixels(
                    self.data[block["offset"]:block["offset"] + block["size"]], pw
                )
                self.assertEqual(len(decoded), pw * ph * 2)

    def test_shared_center_and_distinct_blinks_and_directions(self):
        offsets = character_pack.track_offsets(character_pack.MODEL_TRACK_STEPS)
        frames = self.metadata["frames"]
        center = frames[0]["base"]
        for offset in offsets:
            self.assertEqual(frames[offset]["base"], center)
        self.assertEqual(len({tuple(block.values()) for block in frames[0]["blinks"]}), 4)
        self.assertNotEqual(frames[23]["base"], frames[offsets[1] + 23]["base"])
        self.assertNotEqual(center, frames[offsets[10] + 23]["base"])

    def test_compression_round_trip(self):
        image = generate_solar.Image.new("RGB", (4, 3), (255, 128, 0))
        raw = generate_solar.rgb565(image)
        self.assertEqual(
            sprite_codec.decode_pixels(generate_solar.compress(raw, 4), 4), raw
        )

    def test_foreground_keeps_saturated_yellow_and_background_is_black(self):
        clean, eyes, sun = generate_solar.prepare_artwork()
        image = generate_solar.render(clean, eyes, sun, 0, 0, 0, 24)
        cx, cy, radius = sun
        red, green, blue = image.getpixel((round(cx), round(cy - radius * 0.75)))
        self.assertGreater(red, 160)
        self.assertGreater(green, 140)
        self.assertLess(blue, 140)
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

    def test_solar_levels_are_visually_ordered(self):
        clean, eyes, sun = generate_solar.prepare_artwork()

        def stats(direction):
            count = character_pack.MODEL_TRACK_STEPS[direction]
            pixels = generate_solar.np.asarray(
                generate_solar.render(clean, eyes, sun, direction, count - 1, 0, count),
                dtype=generate_solar.np.float32,
            )
            lit = pixels.max(axis=2) > 8
            light = pixels[lit].mean()
            saturation = (pixels.max(axis=2) - pixels.min(axis=2))[lit].mean()
            return light, saturation, lit.sum(), pixels.sum()

        night, importing, idle, minimum, maximum = (stats(d) for d in (12, 11, 0, 9, 10))
        self.assertLess(night[0], importing[0])
        self.assertLess(importing[1], idle[1])
        self.assertLess(idle[2], minimum[2])
        self.assertLess(minimum[2], maximum[2])
        self.assertLess(idle[3], minimum[3])
        self.assertLess(minimum[3], maximum[3])

    def test_extreme_levels_remain_inside_canvas_and_differ_in_grayscale(self):
        clean, eyes, sun = generate_solar.prepare_artwork()
        rendered = []
        for direction in (12, 11, 0, 9, 10):
            count = character_pack.MODEL_TRACK_STEPS[direction]
            image = generate_solar.render(clean, eyes, sun, direction, count - 1, 0, count)
            rendered.append(image.convert("L").resize((1, 1)).getpixel((0, 0)))
        self.assertEqual(len(set(rendered)), len(rendered))
        self.assertGreaterEqual(rendered[2] - rendered[1], 4)
        self.assertLess(rendered[0], rendered[2])
        self.assertEqual(rendered[-1], max(rendered))
        count = character_pack.MODEL_TRACK_STEPS[10]
        maximum = generate_solar.render(clean, eyes, sun, 10, count - 1, 0, count)
        x0, y0, x1, y1 = maximum.getbbox()
        self.assertGreater(x0, 0)
        self.assertGreater(y0, 0)
        self.assertLess(x1, generate_solar.SIZE[0])
        self.assertLess(y1, generate_solar.SIZE[1])

    def test_blink_patches_reconstruct_rendered_expressions(self):
        clean, eyes, sun = generate_solar.prepare_artwork()
        offsets = character_pack.track_offsets(character_pack.MODEL_TRACK_STEPS)
        for direction, count in enumerate(character_pack.MODEL_TRACK_STEPS):
            frame = self.metadata["frames"][offsets[direction] + count - 1]
            base = frame["base"]
            canvas = sprite_codec.decode_base(
                self.data[base["offset"]:base["offset"] + base["size"]], self.metadata
            )
            for blink in range(5):
                result = bytearray(canvas)
                if blink:
                    block = frame["blinks"][blink - 1]
                    width, height = frame["patchWidth"], frame["patchHeight"]
                    patch = sprite_codec.decode_pixels(
                        self.data[block["offset"]:block["offset"] + block["size"]], width
                    )
                    for row in range(height):
                        offset = ((frame["patchY"] + row) * generate_solar.SIZE[0]
                                  + frame["patchX"]) * 2
                        result[offset:offset + width * 2] = patch[row * width * 2:(row + 1) * width * 2]
                expected = generate_solar.rgb565(
                    generate_solar.render(clean, eyes, sun, direction, count - 1, blink, count)
                )
                self.assertEqual(result, expected, (direction, blink))


if __name__ == "__main__":
    unittest.main()
