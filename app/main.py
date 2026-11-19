import os, re, uuid, json, shutil, subprocess, threading, time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("DATA_DIR", BASE / "data"))
UPLOADS, OUTPUTS = DATA / "uploads", DATA / "outputs"
for p in (UPLOADS, OUTPUTS): p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ClipSpark AI", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE / "app" / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")
JOBS = {}
LOCK = threading.Lock()
ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024

@app.get("/")
def home():
    return FileResponse(BASE / "app" / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "ffmpeg": shutil.which("ffmpeg") is not None,
            "whisper_model": os.getenv("WHISPER_MODEL", "base"),
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))}

@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks,
                     video: UploadFile = File(...),
                     clip_seconds: int = Form(45),
                     language: str = Form("auto"),
                     max_clips: int = Form(3)):
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Please upload MP4, MOV, MKV or WEBM video.")
    if clip_seconds not in (30, 45, 60):
        raise HTTPException(400, "Clip length must be 30, 45 or 60 seconds.")
    if max_clips not in (1, 2, 3, 5):
        raise HTTPException(400, "Choose 1, 2, 3 or 5 clips.")
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{job_id}{suffix}"
    size = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk: break
                size += len(chunk)
                if size > MAX_BYTES:
                    out.close(); dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"Upload exceeds {MAX_BYTES // (1024*1024)} MB limit.")
                out.write(chunk)
    finally:
        await video.close()
    with LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 2,
                        "message": "Upload received. Preparing video…", "clips": [], "error": None}
    background_tasks.add_task(process_job, job_id, dest, clip_seconds, language, max_clips)
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(404, "Job not found. It may have expired after a server restart.")
        return job

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError((p.stderr or "Video processing failed")[-1800:])
    return p.stdout.strip()

def duration_of(path):
    raw = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return max(0.0, float(raw))

def transcribe(path, language):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("Whisper is not installed. Install requirements.txt and restart the server.") from e
    model_name = os.getenv("WHISPER_MODEL", "base")
    model = WhisperModel(model_name, device=os.getenv("WHISPER_DEVICE", "cpu"),
                         compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    lang = None if language == "auto" else language
    segments, info = model.transcribe(str(path), language=lang, vad_filter=True,
                                      word_timestamps=False, beam_size=5)
    rows = []
    for s in segments:
        txt = re.sub(r"\s+", " ", s.text).strip()
        if txt: rows.append({"start": float(s.start), "end": float(s.end), "text": txt})
    if not rows: raise RuntimeError("No clear speech was detected. Try a video with audible dialogue.")
    return rows, info.language

def fallback_rank(rows, total, clip_len, count):
    # Lightweight transcript-based scoring when Gemini is not configured.
    candidates = []
    for i, seg in enumerate(rows):
        text = seg["text"]
        words = len(text.split())
        score = min(words, 24) * 0.12
        if "?" in text: score += 1.4
        if any(w in text.lower() for w in ["but", "because", "secret", "finally", "never", "best", "why", "how", "लेकिन", "क्यों", "फिर", "आखिर"]):
            score += 1.2
        center = (seg["start"] + seg["end"]) / 2
        start = max(0, min(total - clip_len, center - clip_len / 2))
        candidates.append((score, start))
    candidates.sort(reverse=True)
    chosen = []
    for score, start in candidates:
        if all(abs(start - old) > clip_len * 0.65 for old in chosen):
            chosen.append(start)
        if len(chosen) >= count: break
    if not chosen: chosen = [0]
    return sorted(chosen)

def gemini_rank(rows, total, clip_len, count):
    key = os.getenv("GEMINI_API_KEY")
    if not key: return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        compact = [{"start": round(x["start"],1), "end": round(x["end"],1), "text": x["text"][:260]} for x in rows]
        prompt = (
            "You select engaging, self-contained short-video moments from a transcript. "
            "Return ONLY JSON: {\"starts\":[number,...]}. Choose up to "
            f"{count} distinct clip start times in seconds. Each clip is {clip_len} seconds, "
            f"video duration {total:.1f} seconds. Favor strong hooks, useful/funny/emotional "
            "payoffs and coherent context; avoid overlapping moments. Transcript segments: "
            + json.dumps(compact, ensure_ascii=False)
        )
        response = client.models.generate_content(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), contents=prompt)
        match = re.search(r"\{.*\}", response.text or "", re.S)
        if not match: return None
        starts = json.loads(match.group()).get("starts", [])
        result = []
        for x in starts:
            try:
                s = max(0, min(max(0, total-clip_len), float(x)))
                if all(abs(s-y) > clip_len*0.65 for y in result): result.append(s)
            except Exception: pass
        return sorted(result[:count]) or None
    except Exception:
        return None

def ass_time(seconds):
    cs = int(max(0, seconds) * 100)
    h, rem = divmod(cs, 360000); m, rem = divmod(rem, 6000); s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"

def esc_ass(s):
    return s.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", " ")

def write_ass(path, rows, start, end):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,58,&H00FFFFFF,&H000000FF,&H00101010,&H80000000,1,0,1,4,1,2,70,70,170,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for r in rows:
        a, b = max(start, r["start"]), min(end, r["end"])
        if b <= a: continue
        lines.append(f"Dialogue: 0,{ass_time(a-start)},{ass_time(b-start)},Default,,0,0,0,,{esc_ass(r['text'])}\n")
    path.write_text("".join(lines), encoding="utf-8")

def process_job(job_id, source, clip_len, language, count):
    def update(**kw):
        with LOCK: JOBS[job_id].update(kw)
    try:
        update(status="processing", progress=5, message="Checking video…")
        total = duration_of(source)
        if total < 8: raise RuntimeError("Video is too short. Please upload at least 8 seconds.")
        actual_len = min(float(clip_len), total)
        update(progress=12, message="AI is listening to the video…")
        rows, detected = transcribe(source, language)
        update(progress=58, message="Finding the strongest moments…")
        starts = gemini_rank(rows, total, actual_len, count) or fallback_rank(rows, total, actual_len, count)
        results = []
        for idx, start in enumerate(starts, 1):
            end = min(total, start + actual_len)
            ass = OUTPUTS / f"{job_id}_{idx}.ass"
            out = OUTPUTS / f"{job_id}_{idx}.mp4"
            write_ass(ass, rows, start, end)
            update(progress=60 + int(35 * (idx-1)/max(1,len(starts))),
                   message=f"Rendering Short {idx} of {len(starts)}…")
            vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,subtitles='{ass.as_posix()}'"
            run(["ffmpeg", "-y", "-ss", str(start), "-i", str(source), "-t", str(end-start),
                 "-vf", vf, "-c:v", "libx264", "-preset", os.getenv("FFMPEG_PRESET","veryfast"),
                 "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
            results.append({"name": f"Clip {idx}", "url": f"/outputs/{out.name}",
                            "seconds": round(end-start), "start": round(start,1)})
            ass.unlink(missing_ok=True)
        update(status="completed", progress=100, message=f"{len(results)} Shorts are ready.",
               clips=results, language=detected)
    except Exception as e:
        update(status="failed", progress=100, message="Processing failed.", error=str(e))
    finally:
        try: source.unlink(missing_ok=True)
        except Exception: pass

@app.get("/api/download/{filename}")
def download(filename: str):
    if not re.fullmatch(r"[a-f0-9]{12}_\d+\.mp4", filename):
        raise HTTPException(400, "Invalid filename.")
    path = OUTPUTS / filename
    if not path.exists(): raise HTTPException(404, "File not found.")
    return FileResponse(path, media_type="video/mp4", filename=filename)
