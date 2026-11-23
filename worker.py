import os
import json
import subprocess
import requests
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
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

    # 1. Download Live Stream or VOD (captures up to first 60 mins cleanly)
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--live-from-start",
            "--download-sections", "*00:00:00-01:00:00",
            "-o", video_path,
            source_url,
        ],
        check=True,
    )

    # 2. Word-level transcription on free CPU
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(video_path, word_timestamps=True)

    transcript_lines = []
    all_words = []
    for seg in segments:
        transcript_lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text.strip()}")
        if seg.words:
            for w in seg.words:
                all_words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    full_transcript = "\n".join(transcript_lines)

    # 3. Pick the Top 10 Viral Moments using Free Gemini API
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

    # 4. Caption Colors
    color_map = {
        "Bold Yellow": "&H0000FFFF",
        "Neon Green": "&H0000FF00",
        "Clean White": "&H00FFFFFF",
    }
    primary_color = color_map.get(caption_style, "&H0000FFFF")

    # 5. Connect to Free Supabase Storage
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    uploaded_urls = []

    # 6. Crop to 9:16 Vertical & Burn Dynamic Captions
    for idx, clip in enumerate(clips_meta):
        start_t = float(clip["start"])
        end_t = float(clip["end"])
        duration = end_t - start_t
        if duration < 10:
            continue

        srt_path = f"{work_dir}/clip_{idx}.srt"
        out_path = f"{work_dir}/clip_{idx}.mp4"

        clip_words = [w for w in all_words if w["start"] >= start_t and w["end"] <= end_t]
        with open(srt_path, "w", encoding="utf-8") as f:
            chunk_size = 3
            sub_idx = 1
            for i in range(0, len(clip_words), chunk_size):
                chunk = clip_words[i : i + chunk_size]
                c_start = chunk[0]["start"] - start_t
                c_end = chunk[-1]["end"] - start_t
                text = " ".join(w["word"].upper() for w in chunk)
                f.write(f"{sub_idx}\n{format_srt_time(c_start)} --> {format_srt_time(c_end)}\n{text}\n\n")
                sub_idx += 1

        style_str = (
            f"Fontname=Arial,Fontsize=18,Bold=1,PrimaryColour={primary_color},"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=65"
        )
        vf_filter = f"crop=ih*9/16:ih,scale=1080:1920,subtitles={srt_path}:force_style='{style_str}'"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start_t),
                "-t", str(duration),
                "-i", video_path,
                "-vf", vf_filter,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
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

        public_url = supabase.storage.from_("clipforge-videos").get_public_url(remote_path)
        uploaded_urls.append(public_url)

    # 7. Send all 10 finished 9:16 clips back to Base44
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
    
