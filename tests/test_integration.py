"""End-to-end test. Opt in with SUBOCR_INTEGRATION=1; needs macOS, clang and ffmpeg."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "hello.png"


@unittest.skipUnless(os.environ.get("SUBOCR_INTEGRATION") == "1", "set SUBOCR_INTEGRATION=1 to run")
@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        if shutil.which("ffmpeg") is None:  # pragma: no cover
            self.skipTest("ffmpeg not installed")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.video = self.dir / "hello.mp4"
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-loop",
                "1",
                "-i",
                str(FIXTURE),
                "-t",
                "3",
                "-r",
                "10",
                "-pix_fmt",
                "yuv420p",
                str(self.video),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_subocr(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "subocr", str(self.video), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_transcript_and_srt(self):
        out = self.dir / "out"
        result = self.run_subocr("-o", str(out), "--interval", "1", "--srt", "--text", "--force")
        self.assertEqual(result.returncode, 0, result.stderr)
        transcript = (out / "transcript.txt").read_text(encoding="utf-8")
        self.assertIn("HELLO 123", transcript)
        srt = (out / "subtitle.srt").read_text(encoding="utf-8")
        self.assertIn("HELLO 123", srt)
        self.assertIn("-->", srt)
        self.assertFalse((out / "frames").exists())
        data = (out / "subocr.json").read_text(encoding="utf-8")
        self.assertIn('"engine": "Apple Vision"', data)

    def test_keep_frames_and_force(self):
        out = self.dir / "out"
        first = self.run_subocr("-o", str(out), "--interval", "2", "--keep-frames", "--quiet")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue((out / "frames").is_dir())
        self.assertTrue(list((out / "frames").glob("*.png")))
        again = self.run_subocr("-o", str(out), "--interval", "2", "--quiet")
        self.assertEqual(again.returncode, 2)
        self.assertIn("not empty", again.stderr)
        forced = self.run_subocr("-o", str(out), "--interval", "2", "--force", "--quiet")
        self.assertEqual(forced.returncode, 0, forced.stderr)
        self.assertFalse((out / "frames").exists())


if __name__ == "__main__":
    unittest.main()
