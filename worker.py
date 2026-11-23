import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import requests
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "nodejs")
    .pip_install(
        "yt-dlp",
        "faster-whisper",
        "google-genai",
        "supabase",
        "requests",
        "fastapi[standard]",
    )
)

app = modal.App("clipforge-engine", image=image)


def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"


@app.function(
    cpu=4.0,
    memory=4096,
    timeout=1800,
    secrets=[modal.Secret.from_name("clipforge-secrets")],
)
def process_stream_job(job_id: str, source_url: str, caption_style: str, callback_url: str):
    from faster_whisper import WhisperModel
    from google import genai
    from supabase import create_client

    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    video_path = f"{work_dir}/source.mp4"

    # 1. Fast 720p download with Node.js + mobile/web client support
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "--extractor-args", "youtube:player_client=android,web",
            "--live-from-start",
            "--download-sections", "*00:00:00-00:45:00",
            "--merge-output-format", "mp4",
            "-o", video_path,
            source_url,
        ],
        check=True,
    )

    # 2. Fast 4-thread CPU transcription
    model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
    segments, _ = model.transcribe(video_path, word_timestamps=True, beam_size=1)

    transcript_lines = []
    all_words = []
    for seg in segments:
        transcript_lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text.strip()}")
        if seg.words:
            for w in seg.words:
                all_words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    full_transcript = "\n".join(transcript_lines)

    # 3. Pick Top 10 Viral Moments with Free Gemini API
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"""
    You are a viral short-form video editor for top streamers.
    Analyze this transcript and select the TOP 10 funniest, highest-energy, or most engaging moments.
    Each clip MUST be between 25 and 55 seconds long.
    Return ONLY valid JSON as a list of objects with keys: "start" (float seconds), "end" (float seconds), "title" (string).
    Transcript:
    {full_transcript[:45000]}
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    clips_meta = json.loads(response.text)[:10]

    color_map = {
        "Bold Yellow": "&H0000FFFF",
        "Neon Green": "&H0000FF00",
        "Clean White": "&H00FFFFFF",
    }
    primary_color = color_map.get(caption_style, "&H0000FFFF")
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    # 4. Cut, caption & upload all 10 clips IN PARALLEL (4 workers at once)
    def render_and_upload(item):
        idx, clip = item
        start_t = float(clip["start"])
        end_t = float(clip["end"])
        duration = end_t - start_t
        if duration < 10:
            return None

        srt_path = f"{work_dir}/clip_{idx}.srt"
        out_path = f"{work_dir}/clip_{idx}.mp4"

        clip_words = [w for w in all_words if w["start"] >= start_t and w["end"] <= end_t]
        with open(srt_path, "w", encoding="utf-8") as f:
            chunk_size = 3
            sub_idx = 1
            for i in range(0, len(clip_words), chunk_size):
                chunk = clip_words[i : i + chunk_size]
                c_start = chunk["start"] - start_t
                c_end = chunk[-1]["end"] - start_t
                text = " ".join(w["word"].upper() for w in chunk)
                f.write(f"{sub_idx}\n{format_srt_time(c_start)} --> {format_srt_time(c_end)}\n{text}\n\n")
                sub_idx += 1

        style_str = (
            f"Fontname=Arial,Fontsize=18,Bold=1,PrimaryColour={primary_color},"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=65"
        )
        vf_filter = f"crop=ih*9/16:ih,scale=720:1280,subtitles={srt_path}:force_style='{style_str}'"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start_t),
                "-t", str(duration),
                "-i", video_path,
                "-vf", vf_filter,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
                "-c:a", "aac", "-b:a", "128k",
                out_path,
            ],
            check=True,
        )

        remote_path = f"{job_id}/clip_{idx + 1}.mp4"
        with open(out_path, "rb") as video_file:
            supabase.storage.from_("clipforge-videos").upload(
                path=remote_path,
                file=video_file,
                file_options={"content-type": "video/mp4"},
            )

        return supabase.storage.from_("clipforge-videos").get_public_url(remote_path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(render_and_upload, enumerate(clips_meta)))

    uploaded_urls = [u for u in results if u]

    # 5. Send all 10 finished 9:16 clips back to Base44
    requests.post(
        callback_url,
        json={
            "job_id": job_id,
            "clip_urls": uploaded_urls,
            "secret": os.environ["WEBHOOK_SECRET"],
        },
        timeout=30,
    )


@app.function()
@modal.fastapi_endpoint(method="POST")
def trigger_clip_webhook(payload: dict):
    process_stream_job.spawn(
        job_id=payload["job_id"],
        source_url=payload["source_url"],
        caption_style=payload.get("caption_style", "Bold Yellow"),
        callback_url=payload["callback_url"],
    )
    return {"status": "accepted", "job_id": payload["job_id"]}
        
