#!/usr/bin/env python3
"""
S3 → FFmpeg HLS → S3 pipeline
Downloads a video from a source S3 bucket, converts it to HLS with ffmpeg,
then uploads the resulting .m3u8 playlist and .ts segments to a destination bucket.

Encoding settings mirror the TVNA MediaConvert job (640p rendition only):
  - 640×360, H.264 High, QVBR quality 7, max 1 500 kbps, HRD buffer 3 750 kbps
  - 64 kbps AAC HE-v1 (HEV1), 48 kHz, stereo
  - 3-second HLS segments, GOP size 3 s, closed cadence, B-frame refs enabled
  - Input clipping: 00:00:40 → 00:02:50 (mirrors InputClippings in job JSON)

Requirements:
    pip install boto3
    ffmpeg must be installed and on PATH
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"

# Input clipping defaults from TVNA_HLS.json → InputClippings
DEFAULT_CLIP_START = "00:00:40"   # StartTimecode 00:00:40:00
DEFAULT_CLIP_END   = "00:02:50"   # EndTimecode   00:02:50:00


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_s3_client():
    """Build an S3 client using env-var credentials and the default region."""
    session = boto3.Session(region_name=DEFAULT_REGION)
    return session.client("s3")


def download_from_s3(s3, bucket: str, key: str, local_path: Path) -> None:
    log.info("Downloading s3://%s/%s → %s", bucket, key, local_path)
    try:
        s3.download_file(bucket, key, str(local_path))
    except (BotoCoreError, ClientError) as exc:
        log.error("Download failed: %s", exc)
        raise


def _detect_aac_encoder() -> tuple[str, list[str]]:
    """
    Return the best available AAC encoder and its extra flags.
    Prefers libfdk_aac (HE-AAC v1 / HEV1) to match the MediaConvert
    CodecProfile HEV1 setting. Falls back to the built-in aac encoder.
    """
    probe = subprocess.run(
        ["ffmpeg", "-encoders"], capture_output=True, text=True
    )
    if "libfdk_aac" in probe.stdout:
        log.info("Using libfdk_aac encoder (HE-AAC v1 / HEV1)")
        return "libfdk_aac", ["-profile:a", "aac_he"]
    log.warning("libfdk_aac not found – falling back to built-in aac encoder "
                "(output will be AAC-LC, not HEV1; install libfdk_aac for full compliance)")
    return "aac", []


def _probe_fps(input_path: Path) -> float:
    """
    [§7 FIX] Detect the real frame rate of *input_path* via ffprobe, so GOP
    size can be computed correctly instead of assuming a hardcoded 30 fps.
    Falls back to 30.0 if detection fails for any reason.
    """
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        capture_output=True, text=True,
    )
    raw = probe.stdout.strip()  # e.g. "30000/1001" for 29.97 fps
    try:
        if "/" in raw:
            num, den = raw.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(raw)
        log.info("Detected source fps: %.3f", fps)
        return fps
    except (ValueError, ZeroDivisionError):
        log.warning("Could not parse fps from ffprobe output (%r) — defaulting to 30.0", raw)
        return 30.0


def _run_cmd(cmd: list[str], label: str) -> None:
    """Run *cmd*, always printing stderr, raising on non-zero exit."""
    log.info("Running ffmpeg [%s]:\n  %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr.strip():
        log.debug("ffmpeg stderr [%s]:\n%s", label, result.stderr)
    if result.returncode != 0:
        log.error("ffmpeg FAILED [%s] (exit %d):\n%s",
                  label, result.returncode, result.stderr)
        raise RuntimeError(
            f"ffmpeg [{label}] exited with code {result.returncode}"
        )
    log.info("ffmpeg [%s] finished successfully.", label)


def run_ffmpeg(
    input_path: Path,
    output_dir: Path,
    segment_duration: int = 3,
    clip_start: str | None = DEFAULT_CLIP_START,
    clip_end:   str | None = DEFAULT_CLIP_END,
) -> Path:
    """
    Convert *input_path* to a single-rendition HLS stream matching the TVNA
    MediaConvert job (640p output only):

      640×360  max 1 500 kbps  H.264 High  +  64 kbps AAC HE-v1

    Key settings mapped from TVNA_HLS.json
    ───────────────────────────────────────────────────────────────
    JSON field                           ffmpeg flag / value
    ────────────────────────────────     ─────────────────────────
    Width/Height                         -vf scale=640x360
    MaxBitrate 1 500 000                 -maxrate 1500k
    HrdBufferSize 3 750 000              -bufsize 3750k
    HrdBufferInitialFillPercentage 90    -x264-params vbv-init=0.9
    GopSize 3 (SECONDS)                  -g <real_fps * segment_duration>  [§7 FIX]
    GopClosedCadence 1                   -flags +cgop
    GopBReference ENABLED                -b_strategy 2 / -x264-params b-pyramid=strict
    NumberBFramesBetweenReferenceFrames  -bf 5
    NumberReferenceFrames 3              -refs 3
    SceneChangeDetect ENABLED            -sc_threshold 0  [§7 FIX: disabled — was 40]
    DynamicSubGop ADAPTIVE               -x264-params b-adapt=2
    FieldEncoding PAFF                   -x264-params interlaced=0 (PAFF = progressive-adaptive;
                                           libx264 uses progressive by default, which is correct
                                           for PAFF when input is progressive)
    MinIInterval 0                       -keyint_min <same as -g>  [§7 FIX: was 1]
    EntropyEncoding CABAC                -coder 1
    AdaptiveQuantization MEDIUM          -x264-params aq-mode=2
    QvbrQualityLevel 7                   -crf 23  (QVBR≈7 maps to CRF≈23)
    QualityTuningLevel SINGLE_PASS_HQ    -preset slow
    AntiAlias ENABLED                    (enabled by default in libx264)
    Sharpness 100                        -unsharp 5:5:0.8:3:3:0.4
    CodecProfile HIGH                    -profile:v high
    SegmentLength 3                      -hls_time 3
    NameModifier _640p                   output filename suffix
    InputClippings 00:00:40→00:02:50     -ss / -to on input (or omit with --no-clip)
    ───────────────────────────────────────────────────────────────

    [§7 FIX] GOP size (-g / -keyint_min) is now derived from the ACTUAL
    source fps (detected via ffprobe in _probe_fps) instead of a hardcoded
    30 fps assumption, and scene-change keyframe insertion is fully
    disabled (-sc_threshold 0). Together these keep every HLS segment at
    a uniform ~3.000s, fixing the non-uniform segment duration bug where
    scene-change detection was inserting extra keyframes mid-GOP and a
    29.97 vs 30 fps mismatch was drifting segment boundaries over time.

    Returns the path of the rendition playlist  (<stem>_640p.m3u8).
    """
    stem     = input_path.stem
    playlist = output_dir / f"{stem}_640p.m3u8"
    segments = str(output_dir / f"{stem}_640p_%03d.ts")

    aac_enc, aac_extra = _detect_aac_encoder()

    # [§7 FIX] Detect real fps and compute exact GOP size from it, rather
    # than assuming a hardcoded 30 fps (source is actually 29.97 fps).
    real_fps = _probe_fps(input_path)
    gop_size = round(real_fps * segment_duration)   # e.g. 29.97 * 3 ≈ 90
    log.info(
        "Using GOP size %d (fps=%.3f × segment_duration=%ds)",
        gop_size, real_fps, segment_duration,
    )

    # Build the combined x264-params string
    # vbv-init: HrdBufferInitialFillPercentage 90 → 0.9
    x264_params = ":".join([
        "vbv-init=0.9",       # HrdBufferInitialFillPercentage 90
        "b-adapt=2",          # DynamicSubGop ADAPTIVE
        "b-pyramid=strict",   # GopBReference ENABLED (strict = ENABLED in libx264)
    ])

    # Input seeking / clipping flags (placed BEFORE -i for fast seek)
    clip_flags: list[str] = []
    if clip_start:
        clip_flags += ["-ss", clip_start]
    if clip_end:
        clip_flags += ["-to", clip_end]

    cmd = [
        "ffmpeg", "-y",
        *clip_flags,
        "-i", str(input_path),

        # ── Video ─────────────────────────────────────────────────────────
        "-c:v",          "libx264",
        "-profile:v",    "high",           # CodecProfile HIGH
        "-level:v",      "3.1",
        "-preset",       "slow",           # QualityTuningLevel SINGLE_PASS_HQ
        "-crf",          "23",             # QvbrQualityLevel 7 ≈ CRF 23
        "-maxrate",      "1500k",          # MaxBitrate 1 500 000
        "-bufsize",      "3750k",          # HrdBufferSize 3 750 000
        "-vf",           "scale=640:360,unsharp=5:5:0.8:3:3:0.4",
                                           # Width/Height + AntiAlias/Sharpness 100
        "-pix_fmt",      "yuv420p",
        # GOP / keyframe settings  [§7 FIX: dynamic gop_size, sc_threshold disabled]
        "-g",            str(gop_size),    # GopSize 3 s @ REAL detected fps
        "-keyint_min",   str(gop_size),    # force fixed GOP — no early keyframes
        "-sc_threshold", "0",              # disable scene-cut keyframe insertion
        "-bf",           "5",              # NumberBFramesBetweenReferenceFrames 5
        "-refs",         "3",              # NumberReferenceFrames 3
        "-b_strategy",   "2",              # GopBReference ENABLED
        "-flags",        "+cgop",          # GopClosedCadence 1
        # Encoding quality / entropy
        "-coder",        "1",              # EntropyEncoding CABAC
        "-x264-params",  x264_params,

        # ── Audio ──────────────────────────────────────────────────────────
        "-c:a",  aac_enc,
        *aac_extra,                        # -profile:a aac_he if libfdk_aac
        "-b:a",  "64k",                    # Bitrate 64 000
        "-ac",   "2",                      # CodingMode CODING_MODE_2_0
        "-ar",   "48000",                  # SampleRate 48 000

        # ── HLS muxer ──────────────────────────────────────────────────────
        "-f",                    "hls",
        "-hls_time",             str(segment_duration),  # SegmentLength 3
        "-hls_list_size",        "0",
        "-hls_segment_type",     "mpegts",
        "-hls_flags",            "independent_segments",
        "-hls_segment_filename", segments,
        str(playlist),
    ]

    _run_cmd(cmd, "640p")
    log.info("Rendition playlist written: %s", playlist)
    return playlist


def upload_hls_to_s3(s3, output_dir: Path, bucket: str, prefix: str) -> None:
    """
    Upload every file inside *output_dir* to s3://<bucket>/<prefix>/<filename>.
    .m3u8 → application/x-mpegURL
    .ts   → video/MP2T
    """
    content_types = {
        ".m3u8": "application/x-mpegURL",
        ".ts":   "video/MP2T",
    }

    files = sorted(output_dir.iterdir())
    log.info("Uploading %d file(s) to s3://%s/%s/", len(files), bucket, prefix)

    for fpath in files:
        if not fpath.is_file():
            continue
        content_type = content_types.get(fpath.suffix, "application/octet-stream")
        s3_key = f"{prefix.rstrip('/')}/{fpath.name}"

        log.info("  ↑  %s  (%s)", s3_key, content_type)
        try:
            s3.upload_file(
                str(fpath),
                bucket,
                s3_key,
                ExtraArgs={"ContentType": content_type},
            )
        except (BotoCoreError, ClientError) as exc:
            log.error("Upload failed for %s: %s", fpath.name, exc)
            raise

    log.info("All files uploaded successfully.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Download from S3, convert to HLS (640p, TVNA MediaConvert settings), "
            "upload to S3."
        )
    )
    p.add_argument("--src-bucket",  required=True,  help="Source S3 bucket name")
    p.add_argument("--src-key",     required=True,  help="Source S3 object key (e.g. videos/input.mp4)")
    p.add_argument("--dst-bucket",  required=True,  help="Destination S3 bucket name")
    p.add_argument("--dst-prefix",  required=True,  help="Key prefix for HLS output (e.g. hls/my-video)")
    p.add_argument("--keep-tmp",    action="store_true",
                   help="Do not delete the temp directory after upload (useful for debugging)")

    # Clipping options
    clip_group = p.add_mutually_exclusive_group()
    clip_group.add_argument(
        "--no-clip", action="store_true",
        help="Encode the full file; ignore the default InputClippings from the job JSON",
    )
    clip_group.add_argument(
        "--clip-start", default=DEFAULT_CLIP_START, metavar="HH:MM:SS",
        help=f"Start timecode for input clipping (default: {DEFAULT_CLIP_START})",
    )
    p.add_argument(
        "--clip-end", default=DEFAULT_CLIP_END, metavar="HH:MM:SS",
        help=f"End timecode for input clipping (default: {DEFAULT_CLIP_END}). "
             "Ignored when --no-clip is set.",
    )

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Verify ffmpeg is available
    if not shutil.which("ffmpeg"):
        log.error("ffmpeg not found on PATH. Please install it before running this script.")
        sys.exit(1)

    s3 = build_s3_client()

    clip_start = None if args.no_clip else args.clip_start
    clip_end   = None if args.no_clip else args.clip_end

    # Use a temp directory as workspace
    tmp_root = Path(tempfile.mkdtemp(prefix="hls_pipeline_"))
    log.info("Working directory: %s", tmp_root)

    success = False
    try:
        # 1. Download source file
        src_filename = Path(args.src_key).name
        local_input  = tmp_root / src_filename
        download_from_s3(s3, args.src_bucket, args.src_key, local_input)

        # 2. Convert to HLS (640p, TVNA settings)
        hls_dir = tmp_root / "hls_output"
        hls_dir.mkdir()
        run_ffmpeg(local_input, hls_dir, clip_start=clip_start, clip_end=clip_end)

        # 3. Upload HLS output
        upload_hls_to_s3(s3, hls_dir, args.dst_bucket, args.dst_prefix)

        stem = Path(args.src_key).stem
        log.info(
            "Done! Playlist available at: s3://%s/%s/%s_640p.m3u8",
            args.dst_bucket, args.dst_prefix.rstrip("/"), stem,
        )
        success = True

    finally:
        if args.keep_tmp or not success:
            log.info("Temp directory preserved for inspection: %s", tmp_root)
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
            log.info("Temp directory cleaned up.")


if __name__ == "__main__":
    main()
