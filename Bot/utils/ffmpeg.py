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


def check_ffmpeg_installed() -> tuple:
    """Verify ffmpeg/ffprobe are on PATH and runnable. Call once at startup.

    Returns (ok: bool, message: str).
    """
    for binary in ("ffmpeg", "ffprobe"):
        try:
            result = subprocess.run(
                [binary, "-version"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return False, (
                    f"'{binary} -version' exited with code {result.returncode}. "
                    f"Is {binary} installed correctly?"
                )
        except FileNotFoundError:
            return False, (
                f"'{binary}' was not found on PATH. Install FFmpeg "
                f"(e.g. `apt install ffmpeg`) and make sure it's on PATH, "
                f"or check the Dockerfile if running in a container."
            )
        except Exception as e:
            return False, f"Could not verify '{binary}': {e}"
    return True, "ffmpeg and ffprobe are available."


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
        return False, f"Input file is empty (0 bytes) -- likely an incomplete download: {os.path.basename(file_path)}"
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
