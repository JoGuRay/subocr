import json
import tempfile
import unittest
from pathlib import Path

from subocr import cli


def sample(time, lines):
    return {"time": time, "lines": lines, "text": [{"text": line, "confidence": 0.9, "box": [0, 0, 0.1, 0.1]} for line in lines]}


class TimestampTests(unittest.TestCase):
    def test_srt_timestamp(self):
        self.assertEqual(cli.srt_timestamp(0), "00:00:00,000")
        self.assertEqual(cli.srt_timestamp(1.2345), "00:00:01,234")
        self.assertEqual(cli.srt_timestamp(61.5), "00:01:01,500")
        self.assertEqual(cli.srt_timestamp(3661.001), "01:01:01,001")

    def test_human_duration(self):
        self.assertEqual(cli.human_duration(3.2), "3.2s")
        self.assertEqual(cli.human_duration(65), "1m05s")
        self.assertEqual(cli.human_duration(3661), "1h01m01s")


class SamplingTests(unittest.TestCase):
    def test_sample_count(self):
        self.assertEqual(cli.sample_count(3, 1), 3)
        self.assertEqual(cli.sample_count(3, 2), 2)
        self.assertEqual(cli.sample_count(0.5, 1), 1)

    def test_sample_count_rejects_empty_range(self):
        with self.assertRaises(cli.ToolError):
            cli.sample_count(0, 1)

    def test_parse_languages(self):
        self.assertEqual(cli.parse_languages("zh-Hans, en-US"), ["zh-Hans", "en-US"])
        with self.assertRaises(cli.ToolError):
            cli.parse_languages(" , ")


class SampleBuildingTests(unittest.TestCase):
    def test_sorts_boxes_top_down_and_dedupes(self):
        rows = [
            {
                "text": [
                    {"text": "bottom", "confidence": 0.9, "box": [0.1, 0.05, 0.2, 0.1]},
                    {"text": "top", "confidence": 0.8, "box": [0.1, 0.8, 0.2, 0.1]},
                    {"text": "top", "confidence": 0.7, "box": [0.1, 0.8, 0.2, 0.1]},
                    {"text": "   ", "confidence": 0.9, "box": [0.0, 0.0, 0.0, 0.0]},
                ]
            }
        ]
        samples = cli.build_samples(rows, [0.0], 0.0)
        self.assertEqual(samples[0]["lines"], ["top", "bottom"])
        self.assertEqual(len(samples[0]["text"]), 3)
        self.assertAlmostEqual(samples[0]["time"], 0.0, places=3)

    def test_confidence_filter(self):
        rows = [{"text": [{"text": "keep", "confidence": 0.9, "box": [0, 0, 0.1, 0.1]},
                          {"text": "drop", "confidence": 0.1, "box": [0, 0, 0.1, 0.1]}]}]
        samples = cli.build_samples(rows, [0.0], 0.5)
        self.assertEqual(samples[0]["lines"], ["keep"])

    def test_frame_paths_only_when_requested(self):
        rows = [{"text": []}]
        self.assertNotIn("frame", cli.build_samples(rows, [1.0], 0.0)[0])
        with_frames = cli.build_samples(rows, [1.0], 0.0, [Path("/tmp/frames/000001.png")])
        self.assertEqual(with_frames[0]["frame"], "frames/000001.png")


class CueTests(unittest.TestCase):
    def test_merges_identical_consecutive_lines(self):
        samples = [sample(0, ["A"]), sample(1, ["A"]), sample(2, ["A"])]
        cues = cli.build_cues(samples, 1.0, 3.0)
        self.assertEqual(cues, [{"start": 0.0, "end": 2.5, "text": "A"}])

    def test_splits_on_blank_and_change(self):
        samples = [sample(0, ["A"]), sample(1, []), sample(2, ["A"]), sample(3, ["B"])]
        cues = cli.build_cues(samples, 1.0, 4.0)
        self.assertEqual([cue["text"] for cue in cues], ["A", "A", "B"])
        self.assertEqual(len(cues), 3)

    def test_cue_centres_on_sample(self):
        cues = cli.build_cues([sample(1, ["A"])], 2.0, 5.0)
        self.assertEqual(cues, [{"start": 0.0, "end": 2.0, "text": "A"}])

    def test_cue_covers_one_interval_inside_range(self):
        cues = cli.build_cues([sample(1, ["A"])], 2.0, 5.0, range_start=1.0, range_end=5.0)
        self.assertEqual(cues, [{"start": 1.0, "end": 3.0, "text": "A"}])

    def test_clamps_to_range(self):
        cues = cli.build_cues([sample(1, ["A"])], 2.0, 5.0, range_start=1.0, range_end=2.5)
        self.assertEqual(cues, [{"start": 1.0, "end": 2.5, "text": "A"}])

    def test_multiline_text_is_kept(self):
        cues = cli.build_cues([sample(0, ["line one", "line two"])], 1.0, 1.0)
        self.assertEqual(cues[0]["text"], "line one\nline two")

    def test_render_srt(self):
        text = cli.render_srt([{"start": 0.0, "end": 1.0, "text": "hello"}])
        self.assertEqual(text.strip().splitlines(), ["1", "00:00:00,000 --> 00:00:01,000", "hello"])


class OutputTests(unittest.TestCase):
    def test_refuses_non_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "subocr.json").write_text("{}")
            with self.assertRaises(cli.ToolError):
                cli.prepare_output(target, force=False)

    def test_force_clears_only_known_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "subocr.json").write_text("{}")
            (target / "frames").mkdir()
            (target / "frames" / "000001.png").write_bytes(b"x")
            (target / "keep.txt").write_text("keep")
            cli.prepare_output(target, force=True)
            self.assertFalse((target / "subocr.json").exists())
            self.assertFalse((target / "frames").exists())
            self.assertTrue((target / "keep.txt").exists())

    def test_json_helpers_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "value.json"
            cli.write_json(target, {"a": "中文"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"a": "中文"})


class NativeCompileTests(unittest.TestCase):
    @unittest.skipUnless(cli.sys.platform == "darwin", "macOS only")
    def test_native_source_compiles(self):
        import shutil
        import subprocess

        clang = shutil.which("clang")
        self.assertIsNotNone(clang)
        source = Path(cli.__file__).resolve().parent / "native" / "ocr_frames.m"
        result = subprocess.run(
            [
                clang,
                "-fobjc-arc",
                "-fsyntax-only",
                "-mmacosx-version-min=14.0",
                str(source),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
