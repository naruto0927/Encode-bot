import os
import json
import subprocess

from Bot import LOG, HW_ACCEL
from Bot.plugins.database.mongo_db import (
    check_crf_mdb,
    check_vcodec_settings,
    check_preset_settings,
    check_resolution_settings,
    check_audio_type_mdb,
    get_user_field,
)
from Bot.utils.ffmpeg import validate_input_file

# Resolution to FFmpeg scale map
RESOLUTION_MAP = {
    "480p": "640:480",
    "720p": "1280:720",
    "1080p": "1920:1080",
}

# Codec map (software)
VCODEC_MAP = {
    "x264": "libx264",
    "x265": "libx265",
}

# Codec map (nvenc)
NVENC_MAP = {
    "x264": "h264_nvenc",
    "x265": "hevc_nvenc",
}

# Codec map (vaapi)
VAAPI_MAP = {
    "x264": "h264_vaapi",
    "x265": "hevc_vaapi",
}

# Audio bitrate by resolution
AUDIO_BITRATE_MAP = {
    "480p": "64k",
    "720p": "128k",
    "1080p": "256k",
}

# Subtitle codecs that are text-based and can be transcoded into MP4's mov_text.
_TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text"}


class FFmpegBuildError(Exception):
    """Raised when a valid FFmpeg command cannot be constructed (e.g. missing
    input file, unavailable hardware acceleration explicitly requested).
    This is a *pre-flight* error -- distinct from FFmpeg itself failing --
    so callers can show the user a precise reason without ever running ffmpeg.
    """


def _has_audio_stream(input_path: str) -> bool:
    """Check if input file has an audio stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", input_path],
            capture_output=True, text=True, timeout=30,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def _nvenc_usable() -> bool:
    """Best-effort real usability check for NVENC, not just 'is it listed'."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_nvenc" not in result.stdout:
            return False
    except Exception:
        return False
    # A listed encoder still fails at runtime with no GPU/driver present.
    # Cheapest real signal without doing a full encode: an NVIDIA device node
    # or nvidia-smi being present.
    if os.path.exists("/dev/nvidia0"):
        return True
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _vaapi_usable() -> bool:
    """Best-effort real usability check for VAAPI."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_vaapi" not in result.stdout:
            return False
    except Exception:
        return False
    return os.path.exists("/dev/dri/renderD128") or os.path.exists("/dev/dri")


def _detect_hw_accel() -> str:
    """Auto-detect usable hardware acceleration. Falls back to software
    encoding ('none') whenever the requested/detected backend isn't actually
    usable on this host, instead of blindly picking it and letting ffmpeg
    fail later with an opaque error.
    """
    if HW_ACCEL == "none":
        return "none"
    if HW_ACCEL == "nvenc":
        return "nvenc" if _nvenc_usable() else "none"
    if HW_ACCEL == "vaapi":
        return "vaapi" if _vaapi_usable() else "none"

    # auto-detect: try nvenc first, then vaapi, else software
    if _nvenc_usable():
        return "nvenc"
    if _vaapi_usable():
        return "vaapi"
    return "none"


def check_hwaccel_available(mode: str) -> tuple:
    """Used by /hwaccel when a user explicitly requests a mode. Returns
    (ok: bool, reason: str) so the caller can reject an impossible request
    up front rather than letting every subsequent encode fail.
    """
    if mode in ("auto", "none"):
        return True, "ok"
    if mode == "nvenc":
        return (True, "ok") if _nvenc_usable() else (
            False, "NVENC was requested, but no NVIDIA GPU/driver was detected on this host."
        )
    if mode == "vaapi":
        return (True, "ok") if _vaapi_usable() else (
            False, "VAAPI was requested, but no compatible /dev/dri render device was found."
        )
    return False, f"Unknown hardware acceleration mode: {mode}"


def _get_stream_info(input_path: str):
    """Get video/audio/subtitle stream info via ffprobe.

    Returns a dict with "video_streams", "audio_streams", "subtitle_streams"
    (each a list of {"index", "codec", "lang", "title"}), and "ok" (whether
    ffprobe ran successfully at all -- distinct from "found zero streams").
    """
    info = {"video_streams": [], "audio_streams": [], "subtitle_streams": [], "ok": False}
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=index,codec_type,codec_name:stream_tags=language,title",
             "-of", "json", input_path],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout or "{}")
        info["ok"] = True
        for stream in data.get("streams", []):
            tags = stream.get("tags", {})
            codec_type = stream["codec_type"]
            # "index" here is deliberately the TYPE-RELATIVE position (0, 1, 2...
            # within audio/subtitle streams only), NOT ffprobe's global stream
            # index. This matches exactly what "-map 0:a:N" / "-map 0:s:N"
            # expect, and what /tracks accepts -- using the global index here
            # (as the previous version did) could point a track-selection
            # request at a stream that doesn't exist under that "-map 0:a:N"
            # addressing, causing ffmpeg to hard-fail with an invalid stream
            # mapping error.
            if codec_type == "video":
                relative_index = len(info["video_streams"])
            elif codec_type == "audio":
                relative_index = len(info["audio_streams"])
            elif codec_type == "subtitle":
                relative_index = len(info["subtitle_streams"])
            else:
                continue
            entry = {
                "index": relative_index,
                "codec": stream.get("codec_name", "?"),
                "lang": tags.get("language", "?"),
                "title": tags.get("title", ""),
            }
            if codec_type == "video":
                info["video_streams"].append(entry)
            elif codec_type == "audio":
                info["audio_streams"].append(entry)
            elif codec_type == "subtitle":
                info["subtitle_streams"].append(entry)
    except Exception as e:
        LOG.warning(f"Failed to get stream info: {e}")
    return info


def _escape_drawtext(text: str) -> str:
    """Escape text for safe use inside an ffmpeg filtergraph drawtext=text='...'
    value. This is ffmpeg's OWN mini-language, unrelated to shell escaping --
    it's still required even though we no longer go through a shell, because
    the text sits inside the -vf filter string which ffmpeg parses itself.

    Order matters: backslash first, then the filtergraph-special characters.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    return text


async def build_ffmpeg_cmd(user_id: int, input_path: str, progress_file: str,
                            trim_start: str = "", trim_end: str = ""):
    """Build the ffmpeg command based on user's DB settings and the input
    file's ACTUAL streams (probed with ffprobe, not assumed).

    Returns (cmd: list[str], output_path: str, diagnostics: dict, warnings: list[str]).

    Raises FFmpegBuildError for pre-flight problems (missing/empty input,
    unavailable hardware acceleration explicitly requested, no video stream
    found) so the caller can surface a precise reason without ever invoking
    FFmpeg and getting back an opaque failure.
    """
    warnings = []

    # ── 1. Validate input exists and is non-empty before touching it ──
    ok, reason = validate_input_file(input_path)
    if not ok:
        raise FFmpegBuildError(reason)

    # ── 2. Probe actual streams (never assume based on file extension) ──
    streams = _get_stream_info(input_path)
    if not streams["ok"]:
        raise FFmpegBuildError(
            "ffprobe could not read this file -- it may be corrupted, "
            "incomplete, or not a media file at all."
        )
    if not streams["video_streams"]:
        raise FFmpegBuildError("No video stream was found in this file.")

    resolution_key = await check_resolution_settings(user_id) or "480p"
    audio_type = await check_audio_type_mdb(user_id) or "aac"
    preset = await check_preset_settings(user_id) or "fast"
    vcodec_key = await check_vcodec_settings(user_id) or "x264"
    crf = await check_crf_mdb(user_id)
    if crf is None:
        crf = 26

    # ── 3. Hardware acceleration: verify it's really usable ──
    user_hw = await get_user_field(user_id, "hw_accel", "auto")
    if user_hw == "auto":
        hw = _detect_hw_accel()
    elif user_hw == "none":
        hw = "none"
    else:
        ok, reason = check_hwaccel_available(user_hw)
        if not ok:
            # User explicitly asked for hw accel we can't provide -- fail
            # fast with a clear reason instead of a mysterious ffmpeg crash.
            raise FFmpegBuildError(reason)
        hw = user_hw

    if hw == "nvenc":
        vcodec = NVENC_MAP.get(vcodec_key, "h264_nvenc")
        preset_flags = ["-preset", "p4", "-rc", "vbr", "-cq", str(crf)]
    elif hw == "vaapi":
        vcodec = VAAPI_MAP.get(vcodec_key, "h264_vaapi")
        preset_flags = ["-qp", str(crf)]
    else:
        vcodec = VCODEC_MAP.get(vcodec_key, "libx264")
        preset_flags = ["-preset", preset, "-crf", str(crf)]

    resolution = RESOLUTION_MAP.get(resolution_key, "640:480")
    audio_bitrate = AUDIO_BITRATE_MAP.get(resolution_key, "128k")

    ext = ".mkv" if input_path.endswith(".mkv") else ".mp4"
    output_path = input_path.rsplit(".", 1)[0] + f"_encoded{ext}"
    output_is_mp4 = ext == ".mp4"

    # ── 4. Video filters ──
    vf_parts = []
    if resolution_key != "1080p":
        if hw == "vaapi":
            vf_parts.append(f"scale_vaapi=w={resolution.split(':')[0]}:h={resolution.split(':')[1]}")
        else:
            vf_parts.append(f"scale={resolution}")

    watermark_text = await get_user_field(user_id, "watermark_text", "")
    if watermark_text:
        safe_text = _escape_drawtext(watermark_text)
        vf_parts.append(
            f"drawtext=text='{safe_text}':fontsize=18:fontcolor=white@0.5"
            f":x=w-tw-10:y=h-th-10:borderw=1:bordercolor=black@0.3"
        )

    watermark_image = await get_user_field(user_id, "watermark_image", "")
    use_image_overlay = bool(watermark_image and not watermark_text and os.path.exists(watermark_image))

    # ── 5. HW accel input flags ──
    hw_input_args = []
    if hw == "nvenc":
        hw_input_args = ["-hwaccel", "cuda"]
    elif hw == "vaapi":
        hw_input_args = ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi",
                          "-vaapi_device", "/dev/dri/renderD128"]

    # ── 6. Trim (as input-side options, both before -i, so seeking is fast
    #        and -to keeps its normal "absolute position in source" meaning) ──
    trim_args = []
    if trim_start:
        trim_args += ["-ss", str(trim_start)]
    if trim_end:
        trim_args += ["-to", str(trim_end)]

    # ── 7. Audio: validate requested track indices actually exist ──
    has_audio = bool(streams["audio_streams"])
    available_audio_indices = {i for i, _ in enumerate(streams["audio_streams"])}
    strict_flag = ["-strict", "-2"] if audio_type == "opus" else []

    audio_tracks = await get_user_field(user_id, "audio_tracks", "")
    subtitle_tracks = await get_user_field(user_id, "subtitle_tracks", "")

    audio_args = []
    if has_audio:
        if audio_tracks:
            requested = [t.strip() for t in audio_tracks.split(",") if t.strip().isdigit()]
            valid = [t for t in requested if int(t) in available_audio_indices]
            invalid = [t for t in requested if int(t) not in available_audio_indices]
            if invalid:
                warnings.append(
                    f"Requested audio track(s) {', '.join(invalid)} do not exist in this "
                    f"file (only {len(streams['audio_streams'])} audio stream(s) found) -- skipped."
                )
            if not valid:
                # Nothing usable from the selection -- fall back to default
                # mapping instead of building a command with zero mapped audio.
                audio_args = ["-c:a", audio_type, "-b:a", audio_bitrate, "-map", "0:a"]
            else:
                audio_args = ["-c:a", audio_type, "-b:a", audio_bitrate]
                for t in valid:
                    audio_args += ["-map", f"0:a:{t}"]
        else:
            audio_args = ["-c:a", audio_type, "-b:a", audio_bitrate, "-map", "0:a"]
    else:
        audio_args = ["-an"]

    # ── 8. Subtitles: validate indices, and pick a codec the OUTPUT
    #        container can actually hold (this is a common real cause of
    #        "FFmpeg returned an error" -- e.g. copying ASS subs into MP4) ──
    sub_streams = streams["subtitle_streams"]
    available_sub_indices = {i for i, _ in enumerate(sub_streams)}
    sub_args = []

    if sub_streams:
        if subtitle_tracks:
            requested = [t.strip() for t in subtitle_tracks.split(",") if t.strip().isdigit()]
            valid = [t for t in requested if int(t) in available_sub_indices]
            invalid = [t for t in requested if int(t) not in available_sub_indices]
            if invalid:
                warnings.append(
                    f"Requested subtitle track(s) {', '.join(invalid)} do not exist in this "
                    f"file (only {len(sub_streams)} subtitle stream(s) found) -- skipped."
                )
            selected = valid
        else:
            selected = [str(i) for i in range(len(sub_streams))]

        # Drop subtitle codecs that can't be represented in the target
        # container instead of letting ffmpeg hard-fail on the whole job.
        keepable = []
        for t in selected:
            codec = sub_streams[int(t)]["codec"]
            if output_is_mp4 and codec not in _TEXT_SUBTITLE_CODECS:
                warnings.append(
                    f"Subtitle track {t} uses '{codec}', which MP4 cannot contain -- "
                    f"dropped (re-encode to MKV to keep image-based subtitles)."
                )
                continue
            keepable.append(t)

        if keepable:
            sub_codec = "mov_text" if output_is_mp4 else "copy"
            sub_args = ["-c:s", sub_codec]
            for t in keepable:
                sub_args += ["-map", f"0:s:{t}"]

    # ── 9. Assemble the filter graph / overlay, fixing the old bug where
    #        an image-watermark filter_complex output was built but then
    #        never actually mapped (plain "-map 0:v" mapped the RAW stream
    #        instead of the filtered one, so the watermark silently had no
    #        effect, or ffmpeg complained about an unused filtergraph). ──
    video_map_args = []
    filter_args = []

    if use_image_overlay:
        filter_args = ["-i", watermark_image,
                        "-filter_complex", "[0:v][1:v]overlay=W-w-10:H-h-10[vout]"]
        video_map_args = ["-map", "[vout]"]
    elif vf_parts:
        filter_args = ["-vf", ",".join(vf_parts)]
        video_map_args = ["-map", "0:v"]
    else:
        video_map_args = ["-map", "0:v"]

    # ── 10. Assemble the full argument list. NOTE: this is a list, executed
    #         via create_subprocess_exec (no shell), so spaces/unicode/quote
    #         characters in paths or watermark text can never break parsing
    #         or be used for shell injection. ──
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    cmd += hw_input_args
    cmd += trim_args
    cmd += ["-progress", progress_file]
    cmd += ["-i", input_path]
    cmd += filter_args
    cmd += video_map_args
    cmd += ["-vcodec", vcodec]
    cmd += preset_flags
    cmd += ["-metadata", "title=Encoded by Erika-Amano"]
    cmd += ["-metadata:s:v", f"title=Erika - {resolution_key} - {vcodec_key}"]
    cmd += ["-metadata:s:a", "title=Erika-Amano"]
    cmd += audio_args
    cmd += sub_args
    cmd += strict_flag
    cmd += [output_path]

    diagnostics = {
        "input_path": input_path,
        "output_path": output_path,
        "resolution": resolution_key,
        "codec": vcodec,
        "hw_accel": hw,
        "video_streams": len(streams["video_streams"]),
        "audio_streams": len(streams["audio_streams"]),
        "subtitle_streams": len(sub_streams),
    }

    return cmd, output_path, diagnostics, warnings
