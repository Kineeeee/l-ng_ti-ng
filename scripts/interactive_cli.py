import os
import sys
import shutil
import subprocess
import json
import glob

# Ensure backend folder is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.config import (
    TRANSLATION_PROVIDER,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    TTS_VOICE,
    TTS_SPEED,
    TTS_ENGINE,
    MIX_ORIGINAL_AUDIO,
    PITCH_METHOD,
)

STEPS = ["download", "transcribe", "translate", "summarize", "tts", "pitch", "align", "subtitle", "render"]

# Mapping cho lựa chọn TTS engine — dùng chung ở cả start_new_job() và list_and_resume_jobs()
_TTS_ENGINE_MAP = {
    "1": "edge-tts", "2": "gemini", "3": "elevenlabs",
    "edge-tts": "edge-tts", "gemini": "gemini", "elevenlabs": "elevenlabs",
}

_PITCH_METHOD_MAP = {
    "1": "none", "2": "shift", "3": "clone", "4": "hoat_ngon",
    "none": "none", "shift": "shift", "clone": "clone", "hoat_ngon": "hoat_ngon",
    "hoat-ngon": "hoat_ngon",
}


def get_python_executable():
    # Attempt to locate venv python
    venv_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

def check_checkpoint(job_dir, step):
    path = os.path.join(job_dir, f"checkpoint_{step}.json")
    return os.path.exists(path)

def get_job_status(job_dir):
    # Scan backward to find furthest completed step
    for step in reversed(STEPS):
        if check_checkpoint(job_dir, step):
            if step == "render":
                return "Completed"
            idx = STEPS.index(step)
            return f"Next: {STEPS[idx + 1]}"
    return "Next: download"

def scan_existing_jobs():
    output_dir = "output"
    if not os.path.exists(output_dir):
        return []
        
    jobs = []
    # Find all subdirectories in output/
    for name in os.listdir(output_dir):
        job_dir = os.path.join(output_dir, name)
        if os.path.isdir(job_dir):
            dl_checkpoint = os.path.join(job_dir, "checkpoint_download.json")
            if os.path.exists(dl_checkpoint):
                try:
                    with open(dl_checkpoint, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    
                    status = get_job_status(job_dir)
                    jobs.append({
                        "job_id": name,
                        "url": info.get("video_url", "Unknown"),
                        "video_path": info.get("video_path", "Unknown"),
                        "status": status,
                        "mtime": os.path.getmtime(dl_checkpoint)
                    })
                except Exception:
                    pass
                    
    # Sort by modification time, newest first
    jobs.sort(key=lambda x: x["mtime"], reverse=True)
    return jobs

def show_config():
    _DEVICE_LABELS = {
        "auto": "auto (MLX on Apple Silicon, else CPU)",
        "mlx":  "mlx  — Apple GPU (nhanh nhất)",
        "cpu":  "cpu  — faster-whisper CPU",
        "groq": "groq — Groq Cloud API (online)",
    }
    device_display = _DEVICE_LABELS.get(WHISPER_DEVICE.lower(), WHISPER_DEVICE)
    print("\n" + "="*50)
    print("      AutoDub VN - Current Configuration       ")
    print("="*50)
    print(f"  Translation Provider : {TRANSLATION_PROVIDER}")
    print(f"  Whisper Model Size   : {WHISPER_MODEL_SIZE}")
    print(f"  Transcribe Backend   : {device_display}")
    print(f"  TTS Engine           : {TTS_ENGINE}")
    print(f"  TTS Voice            : {TTS_VOICE}")
    print(f"  TTS Speed            : {TTS_SPEED}")
    print(f"  Mix Background Audio : {MIX_ORIGINAL_AUDIO}")
    print(f"  Default Pitch Method : {PITCH_METHOD}")
    print(f"  Summarize & Titles   : Enabled (Step 4 / sau Translate)")
    print("="*50)
    input("\nPress Enter to return to main menu...")

def start_new_job():
    print("\n" + "="*50)
    print("              Start a New Dubbing Job           ")
    print("="*50)
    
    url = input("Enter Video URL (e.g. Douyin, YouTube, TikTok): ").strip()
    if not url:
        print("❌ URL cannot be empty. Returning to menu.")
        input("\nPress Enter to continue...")
        return
        
    lang = input("Source Language (auto / en / zh) [default: auto]: ").strip().lower()
    if not lang:
        lang = "auto"
        
    # --- Transcribe backend ---
    print("\nTranscribe Backend:")
    print("  1. local (auto) — mặc định: MLX/GPU trên Apple Silicon, tự động chọn [nhận Enter]")
    print("  2. mlx           — bắt buộc Apple GPU (m1/m2/m3, nhanh nhất)")
    print("  3. cpu           — faster-whisper CPU (chậm hơn, không cần internet)")
    print("  4. groq          — Groq Cloud API (gần như tức thì, cần internet)")
    _device_map = {"1": "auto", "2": "mlx", "3": "cpu", "4": "groq",
                   "local": "auto", "mlx": "mlx", "cpu": "cpu", "groq": "groq", "": "auto"}
    raw_dev = input("Choose (1-4 or name) [default: 1 / auto]: ").strip().lower()
    transcribe_device = _device_map.get(raw_dev, "auto")
    print(f"  → Using backend: {transcribe_device}")

    interactive = input("Enable Interactive Mode to review quality? (y/n) [default: y]: ").strip().lower()
    interactive_flag = "-i" if interactive != 'n' else ""
    
    trans_method = input("Translation Method: (1) Machine (LLM), (2) Manual [default: 1]: ").strip()
    manual_trans_flag = "--manual-translate" if trans_method == "2" else ""

    print("\nTTS Engine:")
    print("  1. edge-tts    — Giọng Microsoft (miễn phí)")
    print("  2. gemini      — Giọng Google có cảm xúc (miễn phí)")
    print("  3. elevenlabs  — Giọng biểu cảm siêu thực (cần API key)")
    raw_tts = input("Choose (1-3 or name) [default: use .env]: ").strip().lower()
    tts_engine = _TTS_ENGINE_MAP.get(raw_tts, "")
    
    print("\nVoice Pitch Processing Method:")
    print("  1. none      — Giữ nguyên giọng TTS gốc [mặc định / Enter]")
    print("  2. shift     — Thay đổi tông trầm/bổng cho khớp giọng gốc (Simple Pitch Shift)")
    print("  3. clone     — Nhại ngữ điệu nhấn nhá từ giọng gốc (F0 Contour Cloning)")
    print("  4. hoat_ngon — Giọng tươi trẻ, hoạt ngôn (+3.5 semitones, kiểu Cô gái hoạt ngôn)")
    raw_pitch = input("Choose (1-4 or name) [default: 1 / none]: ").strip().lower()
    pitch = _PITCH_METHOD_MAP.get(raw_pitch, "none")
    print(f"  → Pitch method: {pitch}")
        
    subs = input("Include Subtitles? (y/n) [default: y]: ").strip().lower()
    subs_flag = "" if subs != 'n' else "--no-subs"

    summarize = input("Generate Video Summary & Suggested Titles? (y/n) [default: y]: ").strip().lower()
    summarize_flag = "" if summarize != 'n' else "--skip-summarize"
    
    # Construct cli command
    cmd = [
        get_python_executable(),
        os.path.join(os.path.dirname(__file__), "cli_pipeline.py"),
        url,
        "--lang", lang,
        "--pitch", pitch,
        "--transcribe-device", transcribe_device,
    ]
    if tts_engine:
        cmd.extend(["--tts-engine", tts_engine])
    if interactive_flag:
        cmd.append(interactive_flag)
    if manual_trans_flag:
        cmd.append(manual_trans_flag)
    if subs_flag:
        cmd.append(subs_flag)
    if summarize_flag:
        cmd.append(summarize_flag)
        
    print("\n🚀 Starting Pipeline...")
    print(f"Command: {' '.join(cmd)}")
    print("="*50 + "\n")
    
    # Run the command interactively (keep stdin/stdout attached)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline stopped with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
        
    input("\nPress Enter to return to main menu...")

def list_and_resume_jobs():
    while True:
        jobs = scan_existing_jobs()
        print("\n" + "="*60)
        print("                 View & Resume Existing Jobs                 ")
        print("="*60)
        
        if not jobs:
            print("  No previous jobs found in output/ directory.")
            print("="*60)
            input("\nPress Enter to return to main menu...")
            return
            
        print(f"  {'#':<3} | {'Job ID':<10} | {'Status':<18} | {'Video URL'}")
        print("-" * 60)
        for idx, job in enumerate(jobs, start=1):
            url_display = job["url"]
            if len(url_display) > 30:
                url_display = url_display[:27] + "..."
            print(f"  {idx:<3} | {job['job_id']:<10} | {job['status']:<18} | {url_display}")
        print("-" * 60)
        print("  0. Back to Main Menu")
        print("="*60)
        
        choice = input("Select a job to resume or view (0 to cancel): ").strip()
        if choice == '0' or not choice:
            return
            
        try:
            selection_idx = int(choice) - 1
            if selection_idx < 0 or selection_idx >= len(jobs):
                print("❌ Invalid selection.")
                continue
        except ValueError:
            print("❌ Please enter a number.")
            continue
            
        selected_job = jobs[selection_idx]
        job_id = selected_job["job_id"]
        
        # Resume options menu
        while True:
            job_dir = os.path.join("output", job_id)
            summary_path = os.path.join(job_dir, "summary.txt")
            has_summary = os.path.exists(summary_path)

            print("\n" + "="*50)
            print(f"  Job: {job_id}")
            print(f"  URL: {selected_job['url']}")
            print(f"  Current Status: {selected_job['status']}")
            print("="*50)
            print("  1. Resume from next autodetected step")
            print("  2. Force resume from a specific step...")
            print("  3. Delete this job's checkpoints/folder")
            if has_summary:
                print("  4. View Summary & Recommended Titles (summary.txt)")
                print("  5. Cancel / Go back")
            else:
                print("  4. Cancel / Go back")
            print("="*50)
            
            action = input("Select action: ").strip()
            if has_summary and action == '5':
                break
            elif not has_summary and action == '4':
                break
            elif not action:
                break
            
            if has_summary and action == '4':
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        print("\n" + f.read())
                except Exception as e:
                    print(f"❌ Error reading summary file: {e}")
                input("\nPress Enter to return...")
                continue
                
            cmd = [
                get_python_executable(),
                os.path.join(os.path.dirname(__file__), "cli_pipeline.py"),
                "--resume", job_id
            ]
            
            interactive = input("Enable Interactive Mode to review quality? (y/n) [default: y]: ").strip().lower()
            if interactive != 'n':
                cmd.append("-i")
                
            manual_trans = input("Use manual translation instead of machine translation? (y/n) [default: n]: ").strip().lower()
            if manual_trans == 'y':
                cmd.append("--manual-translate")

            print("\nTTS Engine:")
            print("  1. edge-tts    — Giọng Microsoft (miễn phí)")
            print("  2. gemini      — Giọng Google có cảm xúc (miễn phí)")
            print("  3. elevenlabs  — Giọng biểu cảm siêu thực (cần API key)")
            raw_tts = input("Choose (1-3 or name) [default: use .env]: ").strip().lower()
            tts_engine = _TTS_ENGINE_MAP.get(raw_tts, "")
            if tts_engine:
                cmd.extend(["--tts-engine", tts_engine])

            print("\nVoice Pitch Processing Method:")
            print("  1. none      — Giữ nguyên giọng TTS gốc [mặc định / Enter]")
            print("  2. shift     — Thay đổi tông trầm/bổng cho khớp giọng gốc (Simple Pitch Shift)")
            print("  3. clone     — Nhại ngữ điệu nhấn nhá từ giọng gốc (F0 Contour Cloning)")
            print("  4. hoat_ngon — Giọng tươi trẻ, hoạt ngôn (+3.5 semitones, kiểu Cô gái hoạt ngôn)")
            raw_pitch = input("Choose (1-4 or name) [default: use .env/none]: ").strip().lower()
            pitch_method = _PITCH_METHOD_MAP.get(raw_pitch, "")
            if pitch_method:
                cmd.extend(["--pitch", pitch_method])
                
            if action == '1':
                pass
            elif action == '2':
                print("\nSteps:")
                for step in STEPS:
                    print(f"  - {step}")
                resume_step = input("\nEnter step to resume from: ").strip().lower()
                if resume_step not in STEPS:
                    print("❌ Invalid step name. Action cancelled.")
                    continue
                cmd.extend(["--resume-from", resume_step])
                
                use_cache = input(f"\nDo you want to use existing data cache for '{resume_step}' and subsequent steps? (y/n) [default: y]: ").strip().lower()
                if use_cache == 'n':
                    cmd.append("--clear-cache")
            elif action == '3':
                confirm = input(f"⚠️ Are you sure you want to delete job {job_id} and all its output files? (y/n): ").strip().lower()
                if confirm == 'y':
                    job_dir = os.path.join("output", job_id)
                    if os.path.exists(job_dir):
                        shutil.rmtree(job_dir)
                    # Delete output/job_id.mp4 and output/job_id.wav
                    for f in glob.glob(os.path.join("output", f"{job_id}.*")):
                        os.remove(f)
                    print(f"✅ Job {job_id} deleted successfully.")
                    break
                else:
                    continue
            else:
                print("❌ Invalid option.")
                continue
                
            # Launch command
            print("\n🚀 Launching Pipeline...")
            print(f"Command: {' '.join(cmd)}")
            print("="*50 + "\n")
            
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Pipeline exited with error code {e.returncode}")
            except KeyboardInterrupt:
                print("\n\n⚠️ Process interrupted by user.")
                
            input("\nPress Enter to continue...")
            break

def main():
    while True:
        print("\n" + "="*50)
        print("          AutoDub VN - Interactive TUI Menu        ")
        print("==================================================")
        print("  1. Start a New Dubbing Job")
        print("  2. View and Resume Existing Jobs (Checkpoints)")
        print("  3. View Configuration Settings")
        print("  4. Exit")
        print("="*50)
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            start_new_job()
        elif choice == '2':
            list_and_resume_jobs()
        elif choice == '3':
            show_config()
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")
            input("Press Enter to try again...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
