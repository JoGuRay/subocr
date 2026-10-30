#!/usr/bin/env python3
"""subocr: turn a video into local, timestamped OCR text with Apple Vision.

The tool samples frames on a fixed grid, recognises text locally, and writes a
JSON record plus optional SRT subtitles and a plain-text transcript. It never
uploads media, never calls a cloud service, and refuses to overwrite output
unless asked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__

DEFAULT_LANGUAGES = ("zh-Hans", "en-US")
DEFAULT_INTERVAL = 1.0
DEFAULT_MAX_WIDTH = 1280
DEFAULT_MAX_FRAMES = 7200
ARTIFACTS = ("frames", "frames.json", "ocr.raw.json", "subocr.json", "subtitle.srt", "transcript.txt", "ocr.log")


class ToolError(RuntimeError):
    """A recoverable error with a user-facing message."""


def run_tool(command: Sequence[Any], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    argv = [str(part) for part in command]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as error:
        raise ToolError("missing executable: %s" % argv[0]) from error
    except subprocess.TimeoutExpired as error:
        raise ToolError("%s timed out after %ss" % (argv[0], timeout)) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ToolError("%s exited %s: %s" % (argv[0], result.returncode, detail[-2000:]))
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fraction(value: Any) -> Optional[float]:
    """Parse ffprobe rationals such as ``30000/1001``."""
    if value in (None, "", "N/A"):
        return None
    text = str(value)
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            top, bottom = float(numerator), float(denominator)
        except ValueError:
            return None
        return top / bottom if bottom else None
    try:
        return float(text)
    except ValueError:
        return None


def probe_video(video: Path) -> Dict[str, Any]:
    try:
        result = run_tool(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
            timeout=120,
        )
    except ToolError as error:
        raise ToolError("not a readable video: %s" % video) from error
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ToolError("ffprobe returned invalid JSON") from error
    streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"]
    if not streams:
        raise ToolError("input has no video stream: %s" % video)
    stream = streams[0]
    duration = fraction(data.get("format", {}).get("duration")) or fraction(stream.get("duration"))
    if not duration or not math.isfinite(duration) or duration <= 0:
        raise ToolError("cannot determine video duration: %s" % video)
    return {
        "duration": duration,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "codec": stream.get("codec_name"),
        "fps": fraction(stream.get("avg_frame_rate")) or fraction(stream.get("r_frame_rate")),
    }


def sample_count(span: float, interval: float) -> int:
    if span <= 0:
        raise ToolError("empty sampling range")
    if interval <= 0:
        raise ToolError("--interval must be greater than 0")
    return max(1, int(math.ceil(span / interval)))


def prepare_output(out: Path, force: bool) -> Path:
    out = out.resolve()
    if out.exists() and not out.is_dir():
        raise ToolError("output path is not a directory: %s" % out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        if not force:
            raise ToolError("output directory is not empty: %s (use --force)" % out)
        for name in ARTIFACTS:
            target = out / name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
    return out


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_frames(
    video: Path,
    frames_dir: Path,
    interval: float,
    start: float,
    end: float,
    max_width: int,
    limit: int,
    timeout: float,
    source_width: int,
) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=False)
    count = sample_count(end - start, interval)
    if count > limit:
        raise ToolError(
            "sampling would produce %d frames (limit %d); raise --interval or --max-frames" % (count, limit)
        )
    command: List[Any] = ["ffmpeg", "-v", "error", "-nostdin", "-n"]
    if start > 0:
        command += ["-ss", "%.6f" % start]
    command += ["-t", "%.6f" % (end - start)]
    command += ["-i", video, "-map", "0:v:0", "-an"]
    filters = ["fps=%.6f" % (1.0 / interval)]
    if max_width and source_width > max_width:
        width = max_width if max_width % 2 == 0 else max_width - 1
        filters.append("scale=%d:-2" % width)
    command += ["-vf", ",".join(filters), "-frames:v", str(limit)]
    command.append(frames_dir / "%06d.png")
    run_tool(command, timeout=timeout)
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise ToolError("ffmpeg produced no frames; is the range inside the video?")
    return frames


def native_binary() -> Path:
    override = os.environ.get("SUBOCR_NATIVE")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise ToolError("SUBOCR_NATIVE does not point to a file: %s" % path)
        return path
    if sys.platform != "darwin":
        raise ToolError("subocr needs macOS: text recognition uses Apple Vision")
    release = platform.mac_ver()[0]
    if release:
        major, _, _ = release.partition(".")
        if major.isdigit() and int(major) < 14:
            raise ToolError("subocr needs macOS 14 or newer (found %s)" % release)
    source = Path(__file__).resolve().parent / "native" / "ocr_frames.m"
    if not source.is_file():
        raise ToolError("native OCR source is missing: %s" % source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    cache = cache_root / "subocr" / "bin" / digest
    binary = cache / "ocr_frames"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    cache.mkdir(parents=True, exist_ok=True)
    print("building native OCR helper (first run only)", file=sys.stderr)
    temporary = cache / (".ocr_frames.%d" % os.getpid())
    try:
        run_tool(
            [
                "clang",
                "-fobjc-arc",
                "-O2",
                "-mmacosx-version-min=14.0",
                "-framework",
                "Foundation",
                "-framework",
                "Vision",
                "-framework",
                "CoreML",
                source,
                "-o",
                temporary,
            ],
            timeout=180,
        )
        temporary.replace(binary)
    except ToolError as error:
        if temporary.exists():
            temporary.unlink()
        raise ToolError(
            "cannot build the native OCR helper. Install Xcode Command Line Tools "
            "(`xcode-select --install`) and retry.\n%s" % error
        ) from error
    return binary


def run_ocr(
    binary: Path,
    frames: List[Path],
    times: List[float],
    languages: Sequence[str],
    out: Path,
    timeout: float,
) -> List[Dict[str, Any]]:
    frames_json = out / "frames.json"
    raw_json = out / "ocr.raw.json"
    write_json(frames_json, [{"time": times[i], "path": str(frames[i])} for i in range(len(frames))])
    command = [str(binary), str(frames_json), str(raw_json), ",".join(languages)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise ToolError("OCR timed out after %ss; see %s" % (timeout, out / "ocr.log")) from error
    (out / "ocr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise ToolError("OCR failed (exit %s); see %s" % (result.returncode, out / "ocr.log"))
    try:
        rows = json.loads(raw_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolError("cannot read OCR output: %s" % error) from error
    if not isinstance(rows, list) or len(rows) != len(frames):
        raise ToolError("OCR output does not match the sampled frame count")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("text"), list):
            raise ToolError("malformed OCR output at frame %d" % (index + 1))
        expected = str(frames[index])
        if row.get("path") != expected:
            raise ToolError("OCR output frame order mismatch at %d" % (index + 1))
        if abs(float(row.get("time", -1)) - times[index]) > 1e-6:
            raise ToolError("OCR output timestamp mismatch at %d" % (index + 1))
    return rows


def normalize_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_samples(
    rows: List[Dict[str, Any]],
    times: List[float],
    min_confidence: float,
    frames: Optional[List[Path]] = None,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        boxes: List[Dict[str, Any]] = []
        for item in row.get("text", []):
            text = normalize_line(item.get("text"))
            if not text:
                continue
            try:
                confidence = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < min_confidence:
                continue
            box = item.get("box") if isinstance(item.get("box"), list) and len(item["box"]) == 4 else [0, 0, 0, 0]
            boxes.append(
                {
                    "text": text,
                    "confidence": round(confidence, 4),
                    "box": [round(float(value), 6) for value in box],
                }
            )
        # Vision boxes use a bottom-left origin; top edge is y + height.
        boxes.sort(key=lambda item: (-(item["box"][1] + item["box"][3]), item["box"][0]))
        lines: List[str] = []
        for item in boxes:
            if item["text"] not in lines:
                lines.append(item["text"])
        sample: Dict[str, Any] = {"time": round(times[index], 3), "text": boxes, "lines": lines}
        if frames is not None:
            sample["frame"] = "frames/%s" % frames[index].name
        samples.append(sample)
    return samples


def build_cues(
    samples: List[Dict[str, Any]],
    interval: float,
    duration: float,
    range_start: float = 0.0,
    range_end: Optional[float] = None,
    minimum: float = 0.3,
) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for sample in samples:
        text = "\n".join(sample["lines"])
        if not text:
            if current is not None:
                cues.append(current)
                current = None
            continue
        if current is not None and current["text"] == text:
            current["last"] = sample["time"]
            continue
        if current is not None:
            cues.append(current)
        current = {"text": text, "first": sample["time"], "last": sample["time"]}
    if current is not None:
        cues.append(current)

    rendered: List[Dict[str, Any]] = []
    for cue in cues:
        start = max(range_start, cue["first"] - interval / 2.0)
        end = min(duration, range_end if range_end is not None else duration, cue["last"] + interval / 2.0)
        ceiling = range_end if range_end is not None else duration
        if end - start < interval:
            end = min(ceiling, start + interval)
        if end - start < minimum:
            start = max(range_start, end - interval)
        if end <= start:
            continue
        rendered.append({"start": round(start, 3), "end": round(end, 3), "text": cue["text"]})
    return rendered


def srt_timestamp(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return "%02d:%02d:%02d,%03d" % (hours, minutes, secs, milliseconds)


def render_srt(cues: List[Dict[str, Any]]) -> str:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            "%d\n%s --> %s\n%s\n"
            % (index, srt_timestamp(cue["start"]), srt_timestamp(cue["end"]), cue["text"])
        )
    return "\n".join(blocks)


def parse_languages(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ToolError("--lang needs at least one language tag")
    return values


def human_duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return "%dh%02dm%02ds" % (hours, minutes, int(secs))
    if minutes:
        return "%dm%02ds" % (minutes, int(secs))
    return "%.1fs" % seconds


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Hide noisy defaults (None/False) while keeping useful ones."""

    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or ""
        if action.default in (None, False, argparse.SUPPRESS):
            return text
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subocr",
        description="Local OCR for video frames on macOS, powered by Apple Vision.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("-o", "--out", type=Path, help="output directory (default: <video>.subocr)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="seconds between sampled frames")
    parser.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    parser.add_argument("--end", type=float, help="end offset in seconds (default: end of video)")
    parser.add_argument("--lang", default=",".join(DEFAULT_LANGUAGES), help="comma-separated Vision language tags")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH, help="downscale frames wider than this (0 disables)")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="refuse runs that would exceed this many samples")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="drop recognised lines below this confidence")
    parser.add_argument("--srt", action="store_true", help="write subtitle.srt")
    parser.add_argument("--text", action="store_true", help="write transcript.txt")
    parser.add_argument("--keep-frames", action="store_true", help="keep sampled PNG frames")
    parser.add_argument("--force", action="store_true", help="overwrite subocr artifacts in a non-empty output directory")
    parser.add_argument("--hash", action="store_true", help="record the source SHA-256 (slow on large files)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="ffmpeg timeout in seconds")
    parser.add_argument("--ocr-timeout", type=float, default=1800.0, help="overall OCR timeout in seconds")
    parser.add_argument("--json", action="store_true", help="print the result JSON instead of a summary")
    parser.add_argument("--quiet", action="store_true", help="suppress the human summary")
    parser.add_argument("--version", action="version", version="subocr %s" % __version__)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        video = args.video.expanduser().resolve()
        if not video.is_file():
            raise ToolError("input is not a file: %s" % video)
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                raise ToolError("%s not found; install FFmpeg first (brew install ffmpeg)" % tool)
        if args.interval <= 0:
            raise ToolError("--interval must be greater than 0")
        if args.start < 0:
            raise ToolError("--start must not be negative")
        if args.max_frames < 1:
            raise ToolError("--max-frames must be at least 1")
        if args.max_width < 0:
            raise ToolError("--max-width must not be negative")
        if not 0.0 <= args.min_confidence <= 1.0:
            raise ToolError("--min-confidence must be between 0 and 1")

        info = probe_video(video)
        end = info["duration"] if args.end is None else args.end
        if end <= args.start:
            raise ToolError("--end must be greater than --start")
        end = min(end, info["duration"])
        if args.start >= info["duration"]:
            raise ToolError("--start is beyond the end of the video")

        out = prepare_output(args.out or (video.parent / (video.stem + ".subocr")), args.force)
        languages = parse_languages(args.lang)
        count = sample_count(end - args.start, args.interval)
        if count > args.max_frames:
            raise ToolError("sampling would produce %d frames (limit %d)" % (count, args.max_frames))
        if not args.quiet:
            print(
                "sampling %s every %ss (%d frames, %s..%s)"
                % (human_duration(end - args.start), args.interval, count, human_duration(args.start), human_duration(end)),
                file=sys.stderr,
            )

        frames = extract_frames(
            video,
            out / "frames",
            args.interval,
            args.start,
            end,
            args.max_width,
            args.max_frames,
            args.timeout,
            info["width"],
        )
        times = [round(args.start + index * args.interval, 6) for index in range(len(frames))]
        binary = native_binary()
        rows = run_ocr(binary, frames, times, languages, out, args.ocr_timeout)
        samples = build_samples(rows, times, args.min_confidence, frames if args.keep_frames else None)
        cues = build_cues(samples, args.interval, end, args.start, end)

        result: Dict[str, Any] = {
            "tool": "subocr",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": str(video),
                "bytes": video.stat().st_size,
                "duration": round(info["duration"], 3),
                "width": info["width"],
                "height": info["height"],
                "codec": info["codec"],
                "fps": round(info["fps"], 6) if info["fps"] else None,
            },
            "sampling": {
                "interval": args.interval,
                "start": args.start,
                "end": round(end, 3),
                "frames": len(samples),
                "precision": "nearest source frame to each grid point; not word-level timing",
            },
            "ocr": {
                "engine": "Apple Vision",
                "languages": languages,
                "compute": "cpu when available",
            },
            "samples": samples,
            "cues": cues,
            "limits": [
                "Sampling and OCR are not continuous viewing; short or flickering text can be missed.",
                "SRT cues are merged from the sampling grid and are approximate, not forced alignment.",
                "Text is recognised locally; accuracy varies with size, contrast, motion and language.",
            ],
        }
        if args.hash:
            result["source"]["sha256"] = sha256_file(video)

        write_json(out / "subocr.json", result)
        files = ["subocr.json"]
        if args.srt:
            (out / "subtitle.srt").write_text(render_srt(cues), encoding="utf-8")
            files.append("subtitle.srt")
        if args.text:
            transcript = "\n".join(cue["text"] for cue in cues)
            (out / "transcript.txt").write_text(transcript + ("\n" if transcript else ""), encoding="utf-8")
            files.append("transcript.txt")
        files.append("ocr.log")

        if args.keep_frames:
            files.append("frames/")
        else:
            shutil.rmtree(out / "frames", ignore_errors=True)
            for leftover in (out / "frames.json", out / "ocr.raw.json"):
                leftover.unlink(missing_ok=True)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif not args.quiet:
            print("subocr %s" % __version__)
            print("  video    %s (%s, %dx%d)" % (video, human_duration(info["duration"]), info["width"], info["height"]))
            print("  samples  %d every %ss" % (len(samples), args.interval))
            print("  cues     %d" % len(cues))
            print("  output   %s" % out)
            print("  files    %s" % ", ".join(files))
        return 0
    except ToolError as error:
        print("subocr: error: %s" % error, file=sys.stderr)
        return 2
    except OSError as error:
        print("subocr: error: %s" % error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("subocr: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
