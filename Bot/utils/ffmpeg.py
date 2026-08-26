import time
import math
import os
import re
import asyncio
import subprocess

from Bot import LOG

# How much stderr to keep in memory per run (bounded to avoid unbounded growth
# on chatty/broken ffmpeg builds). This is independent of the pipe deadlock fix
# below -- we always *drain* the pipe, we just only *keep* the tail of it.
_STDERR_TAIL_LIMIT = 4000  # characters

# ── Capability cache (populated at startup by check_ffmpeg_capabilities) ──────
_CAPS = {
    "ffmpeg": False,
    "ffprobe": False,
    "libx264": False,
    "drawtext": False,
    "pil_watermark": False,   # True when Pillow is available as drawtext fallback
}


def time_formatter(milliseconds: int) -> str:
    seconds, milliseconds = divmod(int(milliseconds), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    weeks, days = divmod(days, 7)
    parts = []
    if weeks:
        parts.append(f"{weeks}w")
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")
    return ":".join(parts) or "0s"


def humanbytes(size) -> str:
    if size in [None, "", 0]:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            break
        size /= 1024
    return f"{size:.2f} {unit}"


# ─────────────────────────────────────────────────────────────────────────────
# Capability detection
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg_installed() -> tuple:
    """Backward-compat shim — callers that only want a simple ok/message check.

    For the full capability report (drawtext, libx264, …) call
    check_ffmpeg_capabilities() instead. This function calls that and returns
    (ok, summary_message) so existing callers still work.
    """
    caps = check_ffmpeg_capabilities()
    if not caps["ffmpeg"] or not caps["ffprobe"]:
        missing = []
        if not caps["ffmpeg"]:
            missing.append("ffmpeg")
        if not caps["ffprobe"]:
            missing.append("ffprobe")
        return False, f"Required binaries not found: {', '.join(missing)}"
    return True, "ffmpeg and ffprobe are available."


def check_ffmpeg_capabilities() -> dict:
    """Probe FFmpeg at startup and populate the global _CAPS dict.

    Returns the same dict so callers can read it immediately.

    Logs one clear line per capability so the Heroku log makes it obvious
    exactly what works and what doesn't, without having to grep through
    filter/encoder lists manually.
    """
    global _CAPS

    # ── ffmpeg binary ──
    try:
        r = subprocess.run(["ffmpeg", "-version"],
                           capture_output=True, text=True, timeout=15)
        _CAPS["ffmpeg"] = (r.returncode == 0)
    except Exception:
        _CAPS["ffmpeg"] = False

    if _CAPS["ffmpeg"]:
        LOG.info("FFmpeg: available")
    else:
        LOG.critical("FFmpeg: NOT FOUND — encoding will not work")
        return _CAPS

    # ── ffprobe binary ──
    try:
        r = subprocess.run(["ffprobe", "-version"],
                           capture_output=True, text=True, timeout=15)
        _CAPS["ffprobe"] = (r.returncode == 0)
    except Exception:
        _CAPS["ffprobe"] = False

    if _CAPS["ffprobe"]:
        LOG.info("FFprobe: available")
    else:
        LOG.warning("FFprobe: NOT FOUND — stream probing and output validation will be skipped")

    # ── libx264 encoder ──
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=15)
        _CAPS["libx264"] = ("libx264" in r.stdout)
    except Exception:
        _CAPS["libx264"] = False

    if _CAPS["libx264"]:
        LOG.info("libx264: available")
    else:
        LOG.warning("libx264: UNAVAILABLE — x264 encoding will fail; check FFmpeg build")

    # ── drawtext filter ──
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                           capture_output=True, text=True, timeout=15)
        _CAPS["drawtext"] = ("drawtext" in r.stdout)
    except Exception:
        _CAPS["drawtext"] = False

    if _CAPS["drawtext"]:
        LOG.info("drawtext filter: available")
    else:
        LOG.warning(
            "WARNING: FFmpeg drawtext filter is unavailable; "
            "using configured watermark fallback."
        )

    # ── Pillow fallback for watermark ──
    if not _CAPS["drawtext"]:
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: F401
            _CAPS["pil_watermark"] = True
            LOG.info("Pillow watermark fallback: available (will burn text onto frames via post-processing)")
        except ImportError:
            _CAPS["pil_watermark"] = False
            LOG.warning(
                "Pillow is not installed — watermark fallback also unavailable. "
                "Install Pillow (`pip install Pillow`) to restore watermarks on this host."
            )

    return _CAPS


def get_ffmpeg_caps() -> dict:
    """Return the cached capability dict. Safe to call after startup."""
    return _CAPS


# ─────────────────────────────────────────────────────────────────────────────
# PIL watermark fallback
# ─────────────────────────────────────────────────────────────────────────────

def _pil_add_watermark(video_path: str, text: str) -> str | None:
    """Burn a text watermark onto every frame of *video_path* using Pillow +
    FFmpeg's rawvideo pipe (decode → draw → re-mux).

    This is the fallback path used when the FFmpeg build lacks drawtext.
    It is notably slower than drawtext (raw frame pipe round-trip), but
    produces a visually identical result.

    Returns the path to the watermarked video on success, or None on failure.
    The caller is responsible for cleaning up the returned file.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        LOG.error("PIL watermark fallback requested but Pillow is not installed.")
        return None

    import tempfile, shutil

    # ── Probe the input so we know dimensions, fps, pixel format ──
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,pix_fmt",
             "-of", "csv=p=0",
             video_path],
            capture_output=True, text=True, timeout=30,
        )
        parts = probe.stdout.strip().split(",")
        if len(parts) < 4:
            LOG.error(f"PIL watermark: ffprobe parse failed: {probe.stdout!r}")
            return None
        width, height = int(parts[0]), int(parts[1])
        fps_raw = parts[2]   # e.g. "24000/1001"
        pix_fmt = parts[3]   # e.g. "yuv420p"
    except Exception as e:
        LOG.error(f"PIL watermark: probe failed: {e}")
        return None

    # ── Choose a font size relative to the video height ──
    font_size = max(16, height // 30)

    # Try to load a TTF; fall back gracefully to the PIL default bitmap font.
    font = None
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS (dev machines)
    ]:
        if os.path.exists(candidate):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except Exception:
                continue

    if font is None:
        # PIL's built-in bitmap font — always available, smaller/fixed size
        font = ImageFont.load_default()

    # ── Pre-render a watermark RGBA image (transparent background) ──
    # Measure text size with a scratch image
    scratch = Image.new("RGBA", (1, 1))
    draw_scratch = ImageDraw.Draw(scratch)
    try:
        bbox = draw_scratch.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow (< 8.0) — use textsize
        tw, th = draw_scratch.textsize(text, font=font)

    margin = 10
    wm_img = Image.new("RGBA", (tw + margin * 2, th + margin * 2), (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(wm_img)
    # Shadow/border
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        wm_draw.text((margin + dx, margin + dy), text, font=font, fill=(0, 0, 0, 100))
    # Main text (50 % alpha white)
    wm_draw.text((margin, margin), text, font=font, fill=(255, 255, 255, 128))

    wm_w, wm_h = wm_img.size
    x_pos = width - wm_w - 10
    y_pos = height - wm_h - 10

    out_path = video_path.rsplit(".", 1)[0] + "_wm." + (video_path.rsplit(".", 1)[1] if "." in video_path else "mkv")

    LOG.info(f"PIL watermark fallback: stamping '{text}' onto {os.path.basename(video_path)}")

    # ── Decode raw RGB frames from FFmpeg ──
    frame_size = width * height * 3  # RGB24

    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    encode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", fps_raw,
        "-i", "pipe:0",
        # Copy audio from original
        "-i", video_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        "-shortest",
        out_path,
    ]

    try:
        dec_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        enc_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        while True:
            raw = dec_proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = Image.frombytes("RGB", (width, height), raw)
            frame_rgba = frame.convert("RGBA")
            frame_rgba.paste(wm_img, (x_pos, y_pos), wm_img)
            enc_proc.stdin.write(frame_rgba.convert("RGB").tobytes())

        enc_proc.stdin.close()
        dec_proc.stdout.close()

        _, enc_stderr = enc_proc.communicate(timeout=300)
        dec_proc.wait(timeout=60)

        if enc_proc.returncode != 0:
            LOG.error(
                f"PIL watermark encode failed (code {enc_proc.returncode}): "
                f"{enc_stderr.decode(errors='replace')[-1000:]}"
            )
            if os.path.exists(out_path):
                os.remove(out_path)
            return None

        LOG.info(f"PIL watermark fallback: done → {os.path.basename(out_path)}")
        return out_path

    except Exception as e:
        LOG.error(f"PIL watermark fallback error: {e}")
        for p in (dec_proc, enc_proc):
            try:
                p.kill()
            except Exception:
                pass
        if os.path.exists(out_path):
            os.remove(out_path)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Input / output validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_input_file(file_path: str) -> tuple:
    """Confirm the downloaded/input file exists, is a real file, and is not empty.

    Returns (ok: bool, reason: str).
    """
    if not file_path:
        return False, "No input file path was provided."
    if not os.path.exists(file_path):
        return False, f"Input file does not exist: {os.path.basename(file_path)}"
    if not os.path.isfile(file_path):
        return False, f"Input path is not a regular file: {os.path.basename(file_path)}"
    try:
        size = os.path.getsize(file_path)
    except OSError as e:
        return False, f"Could not stat input file: {e}"
    if size == 0:
        return False, (
            f"Input file is empty (0 bytes) -- likely an incomplete download: "
            f"{os.path.basename(file_path)}"
        )
    return True, "ok"


def validate_output_file(output_path: str) -> tuple:
    """Confirm ffmpeg actually produced a usable output file, even if it exited 0.

    Checks existence, non-zero size, and that ffprobe can find a video stream.
    Returns (ok: bool, reason: str).
    """
    if not output_path or not os.path.exists(output_path):
        return False, "Output file was not created."
    try:
        size = os.path.getsize(output_path)
    except OSError as e:
        return False, f"Could not stat output file: {e}"
    if size == 0:
        return False, "Output file is empty (0 bytes)."
    # Minimum sanity: anything under 10 KB is almost certainly a broken mux
    if size < 10_240:
        return False, f"Output file is suspiciously small ({size} bytes) — likely a failed encode."
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                output_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if "video" not in result.stdout:
            return False, "Output file has no readable video stream (ffprobe found none)."
    except Exception as e:
        return False, f"Could not verify output with ffprobe: {e}"
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Frame / duration helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_total_frames(file_path: str) -> int:
    """Get total frames using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        frames = result.stdout.strip()
        if frames.isdigit():
            return int(frames)
    except Exception as e:
        LOG.warning(f"ffprobe frame count failed: {e}")

    # Fallback: estimate from duration and fps
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=duration,r_frame_rate",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            fps_parts = parts[0].split("/")
            fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            duration = float(parts[1])
            return int(fps * duration)
    except Exception:
        pass

    return 0


async def _drain_stream(stream, tail_holder: list):
    """Continuously read an asyncio subprocess stream so its pipe never fills up
    and deadlocks the process, while keeping only the last N chars in memory.

    tail_holder is a 1-element list used as a mutable box so the caller can
    read the accumulated text after this task is awaited/cancelled.
    """
    buf = []
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            try:
                text = chunk.decode(errors="replace")
            except Exception:
                text = ""
            buf.append(text)
            # Keep memory bounded: only retain the tail.
            joined = "".join(buf)
            if len(joined) > _STDERR_TAIL_LIMIT:
                joined = joined[-_STDERR_TAIL_LIMIT:]
                buf = [joined]
    finally:
        tail_holder[0] = "".join(buf)


# ─────────────────────────────────────────────────────────────────────────────
# Main FFmpeg runner
# ─────────────────────────────────────────────────────────────────────────────

async def ffmpeg_progress(cmd, file_path: str, progress_file: str, start_time: float,
                           status_msg, ps_name: str, proc_holder: dict = None):
    """Run FFmpeg and update progress in a Telethon message.

    `cmd` MUST be a list of arguments (e.g. ["ffmpeg", "-y", "-i", path, ...]),
    never a shell string -- this avoids shell-quoting bugs with spaces/unicode/
    special characters in filenames or user-supplied text (e.g. watermark text),
    and avoids shell-injection risk entirely.

    If proc_holder dict is provided, the subprocess is stored under
    proc_holder["process"] so it can be cancelled externally.

    Returns (returncode, stderr_tail) -- stderr_tail is the last portion of
    FFmpeg's stderr output, useful for building a meaningful error message.
    """
    if isinstance(cmd, str):
        # Defensive: this function no longer accepts shell strings.
        raise TypeError("ffmpeg_progress() requires a list of arguments, not a shell string.")

    total_frames = get_total_frames(file_path)
    if total_frames == 0:
        total_frames = 1  # prevent division by zero

    # Create empty progress file
    with open(progress_file, "w") as f:
        pass

    LOG.info(f"Running FFmpeg: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if proc_holder is not None:
        proc_holder["process"] = process

    # Drain stdout/stderr concurrently so a chatty/erroring ffmpeg can never
    # fill the pipe buffer and deadlock (the old code created PIPE but never
    # read it, which both hid every error message AND could hang forever).
    stderr_tail = [""]
    stdout_tail = [""]
    stderr_task = asyncio.create_task(_drain_stream(process.stderr, stderr_tail))
    stdout_task = asyncio.create_task(_drain_stream(process.stdout, stdout_tail))

    last_edit_time = 0

    while process.returncode is None:
        await asyncio.sleep(5)

        try:
            with open(progress_file, "r") as f:
                text = f.read()
        except FileNotFoundError:
            await process.wait()
            break

        frames = re.findall(r"frame=(\d+)", text)
        sizes = re.findall(r"total_size=(\d+)", text)

        if not frames:
            if process.returncode is not None:
                break
            continue

        current_frame = int(frames[-1])
        current_size = int(sizes[-1]) if sizes else 0
        time_diff = time.time() - start_time

        if time_diff == 0:
            continue

        speed = current_frame / time_diff
        if speed == 0:
            continue

        percentage = min(current_frame * 100 / total_frames, 100)
        eta_ms = ((total_frames - current_frame) / speed) * 1000

        filled = math.floor(percentage / 5)
        progress_bar = "█" * filled + "░" * (20 - filled)
        size_text = f"{humanbytes(current_size)} of ~{humanbytes((current_size / max(percentage, 0.01)) * 100)}"
        eta_text = time_formatter(eta_ms)

        # Rate-limit edits to once every 8 seconds
        now = time.time()
        if now - last_edit_time < 8:
            continue
        last_edit_time = now

        try:
            await status_msg.edit(
                f"{ps_name}\n\n"
                f"**[{progress_bar}]** `{percentage:.1f}%`\n\n"
                f"**Size:** `{size_text}`\n"
                f"**ETA:** `{eta_text}`\n"
                f"**Speed:** `{speed:.1f} fps`",
            )
        except Exception as e:
            LOG.warning(f"Error editing FFmpeg status: {e}")

        # Check if process ended
        if process.returncode is not None:
            break

    await process.wait()
    # Make sure the drain tasks finish flushing into stderr_tail/stdout_tail.
    await asyncio.gather(stderr_task, stdout_task, return_exceptions=True)

    returncode = process.returncode
    tail = stderr_tail[0].strip() or stdout_tail[0].strip()

    if returncode != 0:
        LOG.error(
            f"FFmpeg exited with code {returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr tail:\n{tail}"
        )

    return returncode, tail


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

async def generate_sample(input_path: str, duration: int = 30, start_at: str = "00:00:30"):
    """Generate a short sample clip from a video. Returns output path or None."""
    ok, reason = validate_input_file(input_path)
    if not ok:
        LOG.warning(f"Sample generation aborted: {reason}")
        return None

    output = input_path.rsplit(".", 1)[0] + "_sample.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start_at),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
            return output
        LOG.warning(
            f"Sample generation failed (code {proc.returncode}): "
            f"{stderr.decode(errors='replace').strip()[-2000:]}"
        )
    except Exception as e:
        LOG.warning(f"Sample generation error: {e}")
    return None


def get_video_duration(file_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", file_path],
            capture_output=True, text=True, timeout=30,
        )
        val = result.stdout.strip()
        if val:
            return float(val)
    except Exception:
        pass
    return 0.0
