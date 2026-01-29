import base64
import io
import os
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import runpod
from huggingface_hub import snapshot_download
from PIL import Image

WAN_REPO_DIR = os.environ.get("WAN_REPO_DIR", "/workspace/Wan2.2")
WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "/models/Wan2.2-S2V-14B")
WAN_OUTPUT_DIR = os.environ.get("WAN_OUTPUT_DIR", "/outputs")
WAN_MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.2-S2V-14B")
WAN_TASK = os.environ.get("WAN_TASK", "s2v-14B")
WAN_SIZE = os.environ.get("WAN_SIZE", "1024*704")
WAN_T5_DTYPE = os.environ.get("WAN_T5_DTYPE", "bf16")
WAN_DIT_DTYPE = os.environ.get("WAN_DIT_DTYPE", "bf16")
WAN_OFFLOAD = os.environ.get("WAN_OFFLOAD", "true").lower() in {"1", "true", "yes"}
WAN_EXTRA_ARGS = os.environ.get("WAN_EXTRA_ARGS", "")


def _write_b64_to_file(b64_str: str, path: str) -> None:
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


def _parse_size(size: str) -> Tuple[int, int]:
    if "*" in size:
        w, h = size.split("*", 1)
    elif "x" in size.lower():
        w, h = size.lower().split("x", 1)
    else:
        raise ValueError(f"Invalid size format: {size}")
    return int(w.strip()), int(h.strip())


def _write_resized_image(b64_str: str, path: str, size: str) -> None:
    raw = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        target_w, target_h = _parse_size(size)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)
        img.save(path, format="PNG")


def _strip_data_uri(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _ensure_model() -> None:
    if os.path.exists(WAN_CKPT_DIR):
        return
    os.makedirs(WAN_CKPT_DIR, exist_ok=True)
    snapshot_download(
        repo_id=WAN_MODEL_ID,
        local_dir=WAN_CKPT_DIR,
        local_dir_use_symlinks=False,
    )


def _latest_mp4(root: str, start_ts: float) -> Optional[str]:
    latest_path = None
    latest_mtime = start_ts

    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(".mp4"):
                continue
            path = os.path.join(dirpath, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime >= latest_mtime:
                latest_mtime = mtime
                latest_path = path

    return latest_path


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})

    audio_b64 = job_input.get("audio_b64") or job_input.get("audio")
    image_b64 = job_input.get("image_b64") or job_input.get("image")
    prompt = job_input.get("prompt", "")
    size = job_input.get("size", WAN_SIZE)

    if not audio_b64 or not image_b64:
        return {"error": "audio_b64 and image_b64 are required"}

    audio_b64 = _strip_data_uri(audio_b64)
    image_b64 = _strip_data_uri(image_b64)

    os.makedirs(WAN_OUTPUT_DIR, exist_ok=True)
    input_dir = os.path.join(WAN_OUTPUT_DIR, "inputs")
    os.makedirs(input_dir, exist_ok=True)

    audio_path = os.path.join(input_dir, "input_audio.wav")
    image_path = os.path.join(input_dir, "input_image.png")

    _write_b64_to_file(audio_b64, audio_path)
    _write_resized_image(image_b64, image_path, size)

    _ensure_model()

    cmd = [
        "python",
        os.path.join(WAN_REPO_DIR, "generate.py"),
        "--task",
        WAN_TASK,
        "--size",
        size,
        "--ckpt_dir",
        WAN_CKPT_DIR,
        "--prompt",
        prompt,
        "--image",
        image_path,
        "--audio",
        audio_path,
        "--t5_fsdp",
        "--t5_cpu",
        "--t5_dtype",
        WAN_T5_DTYPE,
        "--dit_fsdp",
        "--dit_cpu",
        "--dit_dtype",
        WAN_DIT_DTYPE,
    ]

    if WAN_OFFLOAD:
        cmd.append("--offload_model")

    if WAN_EXTRA_ARGS:
        cmd.extend(WAN_EXTRA_ARGS.split())

    start_ts = time.time()
    subprocess.run(cmd, check=True, cwd=WAN_REPO_DIR)

    output_path = _latest_mp4(WAN_REPO_DIR, start_ts) or _latest_mp4(WAN_OUTPUT_DIR, start_ts)
    if not output_path:
        return {"error": "No output video found"}

    with open(output_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("ascii")

    return {
        "video_b64": video_b64,
        "filename": os.path.basename(output_path),
    }


runpod.serverless.start({"handler": handler})
