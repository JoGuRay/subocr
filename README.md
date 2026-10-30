# subocr

Local OCR for video frames on macOS, powered by Apple Vision.

`subocr` samples a video on a fixed grid, recognises the text in each frame with
Apple's on-device Vision framework, and writes timestamped JSON plus optional
SRT subtitles and a plain-text transcript. No cloud service, no model download,
no upload.

```console
$ subocr talk.mp4 --srt --text
subocr 0.1.0
  video    /Users/me/talk.mp4 (12m04s, 1920x1080)
  samples  725 every 1.0s
  cues     96
  output   /Users/me/talk.subocr
  files    subocr.json, subtitle.srt, transcript.txt, ocr.log
```

**中文速览**：给一段视频，按固定间隔抽帧，用 macOS 自带 Vision 在本地识别文字，
输出带时间戳的 JSON，可选 SRT 字幕和纯文本。全程离线、不装模型、不上传。

## Why another OCR tool?

- **Runs where the video is.** No API key, no queue, no upload of private
  meetings, courses or unreleased footage.
- **Native and small.** A small Python package plus a ~180-line Objective-C helper that
  is compiled on first use with the system `clang`. No Python dependencies.
- **Timestamped records, not just text.** Every sample keeps its time, every
  recognised line keeps its confidence and normalised bounding box.
- **Honest about its precision.** Sampling is not word-level timing; the SRT is
  a heuristic merge of the sample grid, and the docs say so.

Typical uses: extracting on-screen text from screen recordings, making rough
subtitles for clips that have burned-in text, auditing what a video actually
says, or turning a folder of talks into searchable text.

## Requirements

- macOS 14 or newer (Apple Silicon or Intel)
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on `PATH`)
- Xcode Command Line Tools (`xcode-select --install`) for the native helper

## Install

```bash
git clone https://github.com/JoGuRay/subocr.git
cd subocr
python3 -m pip install --user .
# or run without installing:
python3 -m subocr --help
```

With [pipx](https://pipx.pypa.io/):

```bash
pipx install git+https://github.com/JoGuRay/subocr.git
```

The first run compiles `subocr/native/ocr_frames.m` into
`~/.cache/subocr/bin/<source-hash>/ocr_frames` (about 10–20 seconds). Later runs
reuse the binary; delete the cache directory to force a rebuild.

## Usage

```bash
# Default: one sample per second, output next to the video in <name>.subocr/
subocr video.mp4

# Half-second sampling, with subtitles and a transcript
subocr video.mp4 --interval 0.5 --srt --text

# Only a section, Chinese only, machine-readable result on stdout
subocr talk.mp4 --start 600 --end 900 --lang zh-Hans --json

# Keep the sampled PNGs for manual review
subocr video.mp4 --keep-frames
```

Run `subocr --help` for every option.

| Option | Default | Meaning |
| --- | --- | --- |
| `-o, --out DIR` | `<video>.subocr` | Output directory |
| `--interval SECONDS` | `1.0` | Time between sampled frames |
| `--start` / `--end` | whole video | Sampling window |
| `--lang` | `zh-Hans,en-US` | Comma-separated Vision language tags |
| `--max-width` | `1280` | Downscale wider frames before OCR (`0` disables) |
| `--max-frames` | `7200` | Refuse runs that would exceed this many samples |
| `--min-confidence` | `0.0` | Drop lines below this confidence |
| `--srt` / `--text` | off | Write `subtitle.srt` / `transcript.txt` |
| `--keep-frames` | off | Keep `frames/*.png` |
| `--hash` | off | Record the source SHA-256 |
| `--json` | off | Print the result JSON instead of a summary |
| `--force` | off | Replace known artifacts in a non-empty output directory |

## Output

```
video.subocr/
├── subocr.json       # source info, samples, cues, limits
├── subtitle.srt      # only with --srt
├── transcript.txt    # only with --text
├── ocr.log           # native helper progress and errors
└── frames/           # only with --keep-frames
```

`subocr.json`:

```json
{
  "source": {"duration": 12.04, "width": 1920, "height": 1080, "fps": 25.0},
  "sampling": {"interval": 1.0, "frames": 725, "precision": "nearest source frame to each grid point; not word-level timing"},
  "samples": [
    {"time": 0.0, "lines": ["Quarterly review"],
     "text": [{"text": "Quarterly review", "confidence": 0.5, "box": [0.12, 0.81, 0.55, 0.07]}]}
  ],
  "cues": [{"start": 0.0, "end": 1.5, "text": "Quarterly review"}]
}
```

Boxes are normalised Vision coordinates with a bottom-left origin.

## How it works

1. `ffprobe` reads duration and dimensions.
2. `ffmpeg` samples the window with the `fps` filter, downscaling anything wider
   than `--max-width` (smaller frames are much faster to recognise).
3. A small Objective-C helper compiles once, then asks Vision to recognise text
   in each PNG at the accurate level, with language correction disabled. Every
   supported compute stage is pinned to the CPU so repeated runs stay stable.
4. Lines are sorted top-to-bottom, deduplicated within a frame, and merged into
   cues when consecutive samples carry the same text.
5. `subocr.json` is written; SRT times are padded by half the sampling interval
   and clamped to the requested window.

## Limits

- Sampling can miss short, flickering or fast-moving text. Use `--interval 0.5`
  or lower for dense material.
- SRT cues are approximate. This is not forced alignment and not a replacement
  for human transcription when exact wording or timing matters.
- Recognition quality depends on text size, contrast, motion and language;
  decorative fonts, heavy compression and overlapping text will produce errors.
- Audio is ignored entirely. There is no speech-to-text and no speaker diarisation.
- macOS-only: the helper uses Vision and CoreML frameworks.

## Privacy

- No network calls of any kind; the only external processes are `ffprobe`,
  `ffmpeg` and `clang`.
- Frames are deleted after OCR unless `--keep-frames` is set.
- Output directories are refused when non-empty unless `--force` is passed, and
  `--force` only replaces artifacts this tool owns.

## Development

```bash
python3 -m unittest discover -s tests -v

# End-to-end test (macOS + ffmpeg; builds a 3s clip from tests/fixtures/hello.png)
SUBOCR_INTEGRATION=1 python3 -m unittest discover -s tests -p 'test_integration.py' -v
```

## License

MIT. See [LICENSE](LICENSE).

The native helper was extracted from a private video-production workflow and
generalised for this repository. The sampling and cue-merging logic is a
reimplementation of the same workflow with stricter validation and no
project-specific paths, schemas or secrets.
