# ClipSpark AI — runnable MVP

A self-hostable web app that accepts an uploaded video, transcribes speech with Faster-Whisper, ranks short moments with Gemini when configured (otherwise uses a transcript heuristic), renders vertical 9:16 MP4 clips with burned-in captions using FFmpeg, and provides downloads.

## Important limits

- This MVP processes **uploaded files**. It does not download arbitrary YouTube videos. Use your own video file or content you have permission to edit. YouTube links are not a reliable/legal substitute for an authorized source file.
- No software can guarantee zero bugs or uninterrupted hosting. Test with a short, non-sensitive video before public launch.
- This is a single-process MVP: job state is in memory and files are stored locally. For a public multi-user service, add a persistent queue/database, object storage, authentication, rate limits, cleanup policy, monitoring and a paid worker plan.
- Gemini is optional. Without `GEMINI_API_KEY`, selection uses a basic transcript heuristic. Whisper still performs transcription locally.
- First run downloads the selected Whisper model and may be slow. CPU processing can take several minutes per video.

## Run locally (Python 3.11 recommended)

1. Install FFmpeg and make sure `ffmpeg` and `ffprobe` are on PATH.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env`, then optionally add a Gemini API key. Export the values in your shell or use a dotenv runner.
5. Start: `uvicorn app.main:app --host 127.0.0.1 --port 8000`
6. Open http://127.0.0.1:8000

## Run with Docker

```bash
docker build -t clipspark .
docker run --rm -p 8000:8000 -v clipspark-data:/srv/data --env-file .env clipspark
```

Open http://localhost:8000.

## Deploy

Deploy the Dockerfile to a container host that supports background CPU work and persistent volumes. Attach a persistent volume at `/srv/data`, set environment variables from `.env.example`, and set a request timeout appropriate for video uploads. Do not commit `.env` or expose API keys in frontend code. Free serverless tiers often time out or lack enough CPU/RAM for Whisper/FFmpeg.

## API

- `GET /api/health` — readiness and configuration check.
- `POST /api/jobs` — multipart fields: `video`, `clip_seconds` (30/45/60), `language` (`auto`, `hi`, `en`, `es`), `max_clips` (1/2/3/5).
- `GET /api/jobs/{job_id}` — processing status and clip URLs.
- `GET /api/download/{filename}` — MP4 download.

## Production checklist

Before opening this to the public:
- Put it behind HTTPS and a reverse proxy with upload limits.
- Add login, per-user quotas, rate limiting and abuse protection.
- Move jobs to Redis/Celery or another durable queue; persist job metadata in a database.
- Store outputs in S3-compatible object storage; expire uploads and outputs automatically.
- Add virus/malware scanning and content moderation appropriate to your use case.
- Add billing/usage limits if processing is paid.
- Test Hindi/English/Hinglish captions and crop behavior on a variety of footage.
