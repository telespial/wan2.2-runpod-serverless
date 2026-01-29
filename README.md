# Runpod Serverless: Wan2.2 S2V (Audio + Image → Video)

This folder contains a minimal Runpod serverless container for Wan2.2 S2V (speech-to-video).
It accepts base64 audio + image, runs `generate.py`, and returns a base64 MP4.

## Files
- `Dockerfile` container build
- `handler.py` Runpod serverless handler
- `requirements.txt` extra deps (runpod, huggingface_hub)

## Environment variables (container)
- `WAN_REPO_DIR` default `/workspace/Wan2.2`
- `WAN_MODEL_ID` default `Wan-AI/Wan2.2-S2V-14B`
- `WAN_CKPT_DIR` default `/models/Wan2.2-S2V-14B`
- `WAN_OUTPUT_DIR` default `/outputs`
- `WAN_TASK` default `s2v-14B`
- `WAN_SIZE` default `1024*704`
- `WAN_OFFLOAD` default `true`
- `WAN_EXTRA_ARGS` optional extra CLI flags

## Build & push (example)
Replace `your-dockerhub-user` and tag as you like.

```bash
docker build -t your-dockerhub-user/wan22-s2v-runpod:latest backend/runpod_wan22_serverless
docker push your-dockerhub-user/wan22-s2v-runpod:latest
```

## Runpod Serverless setup (high level)
1. Create a **Serverless Endpoint** in Runpod.
2. Use **Custom Container** and paste your Docker image.
3. Select a GPU type with enough VRAM for Wan2.2 S2V.
4. Deploy and copy the **Endpoint ID** + **API key**.

## Input schema
This handler expects:

```json
{
  "input": {
    "audio_b64": "<base64-wav>",
    "image_b64": "<base64-png>",
    "prompt": "optional text prompt",
    "size": "1024*704"
  }
}
```

## Output schema

```json
{
  "output": {
    "video_b64": "<base64-mp4>",
    "filename": "..."
  }
}
```

## Notes
- First run may download model weights (large), so expect a long cold start.
- Returning a base64 MP4 can be large; consider uploading to S3/R2 for big outputs.
