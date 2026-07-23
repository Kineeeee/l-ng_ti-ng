import os
import sys
import shutil
import subprocess
import json
import glob

# Force unbuffered stdout/stderr so prompts always appear before input() blocks
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure backend folder is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.config import (
    TRANSLATION_PROVIDER,
    TRANSLATION_STYLE,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    TTS_VOICE,
    TTS_SPEED,
    TTS_ENGINE,
    MIX_ORIGINAL_AUDIO,
    PITCH_METHOD,
    ENABLE_OCR_SUBTITLE,
    OCR_CROP_BOTTOM_RATIO,
    OCR_SAMPLE_FPS,
    OCR_MIN_CONFIDENCE,
    VIDEO_SPEED_FACTOR,
    DUCK_VOLUME_DB,
    TTS_VOLUME_DB,
    AUDIO_SYNC_OFFSET_MS,
    MASK_OLD_SUBS,
    MASK_SUB_COLOR,
    LOGO_PATH,
    LOGO_POSITION,
    WATERMARK_TEXT,
    WATERMARK_POSITION,
    MIRROR_VIDEO,
    MASK_TOP_TEXT,
    TOP_TEXT,
)

STEPS = ["download", "transcribe", "translate", "summarize", "tts", "pitch", "align", "subtitle", "render"]

_TTS_ENGINE_MAP = {
    "1": "edge-tts", "2": "gemini", "3": "elevenlabs",
    "edge-tts": "edge-tts", "gemini": "gemini", "elevenlabs": "elevenlabs",
}

_PITCH_METHOD_MAP = {
    "1": "none", "2": "shift", "3": "clone", "4": "hoat_ngon",
    "none": "none", "shift": "shift", "clone": "clone", "hoat_ngon": "hoat_ngon",
    "hoat-ngon": "hoat_ngon",
}

_TRANSLATION_STYLE_MAP = {
    "1": "standard", "2": "humorous", "3": "storyteller",
    "standard": "standard", "humorous": "humorous", "storyteller": "storyteller",
}


def get_python_executable():
    venv_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

def check_checkpoint(job_dir, step):
    path = os.path.join(job_dir, f"checkpoint_{step}.json")
    return os.path.exists(path)

def get_job_status(job_dir):
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
                    
    jobs.sort(key=lambda x: x["mtime"], reverse=True)
    return jobs

def show_config():
    _DEVICE_LABELS = {
        "auto": "auto (MLX on Apple Silicon GPU, else CPU)",
        "mlx":  "mlx  — Apple GPU (nhanh nhất)",
        "cpu":  "cpu  — faster-whisper CPU",
        "groq": "groq — Groq Cloud API (online)",
    }
    device_display = _DEVICE_LABELS.get(WHISPER_DEVICE.lower(), WHISPER_DEVICE)
    ocr_status = f"Enabled (RapidOCR GPU | Crop {OCR_CROP_BOTTOM_RATIO*100:.0f}% bottom | {OCR_SAMPLE_FPS} FPS)" if ENABLE_OCR_SUBTITLE else "Disabled"
    top_text_display = f"'{TOP_TEXT}'" if TOP_TEXT else "None"
    logo_display = f"{LOGO_PATH} ({LOGO_POSITION})" if LOGO_PATH else "None"
    wm_display = f"'{WATERMARK_TEXT}' ({WATERMARK_POSITION})" if WATERMARK_TEXT else "None"

    print("\n" + "="*68)
    print("        ⚙️ AutoDub VN - Cấu hình Hệ thống Đầy đủ (.env)        ")
    print("="*68)
    print(" 🎬 1. DOWNLOAD & TỐC ĐỘ VIDEO:")
    print(f"    • Video Speed Slowdown   : {VIDEO_SPEED_FACTOR}x (Giảm còn {VIDEO_SPEED_FACTOR*100:.0f}% tốc độ gốc)")
    print()
    print(" 🎙️ 2. TRANSCRIBE & VIDEO OCR:")
    print(f"    • Whisper Model Size     : {WHISPER_MODEL_SIZE}")
    print(f"    • Transcribe Backend     : {device_display}")
    print(f"    • Video OCR Hardsub      : {ocr_status}")
    print(f"    • OCR Min Confidence     : {OCR_MIN_CONFIDENCE}")
    print()
    print(" 🌐 3. DỊCH THUẬT & PHONG CÁCH:")
    print(f"    • Translation Provider   : {TRANSLATION_PROVIDER}")
    print(f"    • Translation Style      : {TRANSLATION_STYLE}")
    print()
    print(" 🔊 4. PHÁT ÂM & LỒNG TIẾNG (TTS):")
    print(f"    • TTS Engine / Voice     : {TTS_ENGINE} ({TTS_VOICE})")
    print(f"    • TTS Speed Adjustment   : {TTS_SPEED}")
    print(f"    • Pitch Processing       : {PITCH_METHOD}")
    print()
    print(" 🎛️ 5. HÒA ÂM & ÂM LƯỢNG (AUDIO MIXING):")
    print(f"    • Mix Original Audio     : {MIX_ORIGINAL_AUDIO}")
    print(f"    • Background Ducking     : {DUCK_VOLUME_DB} dB")
    print(f"    • TTS Volume Boost       : +{TTS_VOLUME_DB} dB")
    print(f"    • Audio Sync Offset      : {AUDIO_SYNC_OFFSET_MS} ms")
    print()
    print(" 🎨 6. ĐỒ HỌA & CHE PHỤ ĐỀ (MASK & WATERMARK):")
    print(f"    • Mask Old Subtitles     : {MASK_OLD_SUBS} (Style: {MASK_SUB_COLOR})")
    print(f"    • Mask Top Banner Box    : {MASK_TOP_TEXT}")
    print(f"    • Top Banner Content     : {top_text_display}")
    print(f"    • Logo Image & Position  : {logo_display}")
    print(f"    • Text Watermark         : {wm_display}")
    print(f"    • Mirror Video (Horizontal): {MIRROR_VIDEO}")
    print("="*68)
    input("\nẤn Enter để quay lại menu chính...")

def start_new_job(initial_url: str = None):
    print("\n" + "="*55)
    print("              🚀 Khởi Tạo Dubbing Job Mới              ")
    print("="*55)
    
    if initial_url:
        url = initial_url.strip()
        print(f"Video URL: {url}")
    else:
        url = input("Dán Video URL (Douyin, YouTube, TikTok,...): ").strip()
        if not url:
            print("❌ URL không được để trống. Quay lại menu.")
            input("\nẤn Enter để tiếp tục...")
            return

    # Mode selection
    print("\nChọn Chế độ Chạy:")
    print("  1. ⚡ Quick Auto (1-Click) — Dùng cấu hình tối ưu sẵn [Ấn Enter]")
    print("  2. 🎛️ Custom Config      — Tùy chỉnh chi tiết (Whisper, TTS, Pitch, Style...)")
    mode_choice = input("\nChọn (1-2) [mặc định: 1 / Quick Auto]: ").strip()

    if mode_choice == "2":
        # Custom Mode
        lang = input("\nNgôn ngữ gốc (auto / en / zh) [mặc định: auto]: ").strip().lower() or "auto"
        
        print("\nTranscribe Backend:")
        print("  1. local (auto) — Mặc định: Apple GPU (MLX) / CPU [nhận Enter]")
        print("  2. mlx           — Ép dùng Apple GPU (M1/M2/M3/M4)")
        print("  3. cpu           — CPU faster-whisper")
        print("  4. groq          — Groq Cloud API (tốc độ cao)")
        _device_map = {"1": "auto", "2": "mlx", "3": "cpu", "4": "groq", "": "auto"}
        raw_dev = input("Chọn (1-4) [mặc định: 1]: ").strip().lower()
        transcribe_device = _device_map.get(raw_dev, "auto")

        interactive = input("\nBật chế độ Xem & Duyệt chất lượng ở các bước? (y/n) [mặc định: y]: ").strip().lower()
        interactive_flag = "-i" if interactive != 'n' else ""

        ocr_choice = input("Bật trích xuất Phụ đề OCR từ Video? (y/n) [mặc định: y]: ").strip().lower()
        no_ocr_flag = "--no-ocr" if ocr_choice == 'n' else ""

        trans_method = input("\nPhương thức dịch: (1) Máy (LLM), (2) Thủ công [mặc định: 1]: ").strip()
        manual_trans_flag = "--manual-translate" if trans_method == "2" else ""

        trans_style = "standard"
        if trans_method != "2":
            print("\nPhong cách dịch thuật:")
            print("  1. standard    — Thuyết minh chuẩn mực, tự nhiên [nhận Enter]")
            print("  2. humorous    — Hài hước & Bình luận dí dỏm (Fair Use)")
            print("  3. storyteller — Kể chuyện kịch tính, TikTok viral")
            raw_style = input("Chọn (1-3) [mặc định: 1]: ").strip().lower()
            trans_style = _TRANSLATION_STYLE_MAP.get(raw_style, "standard")

        print("\nTTS Engine:")
        print("  1. edge-tts    — Microsoft (Miễn phí)")
        print("  2. gemini      — Google (Cảm xúc)")
        print("  3. elevenlabs  — ElevenLabs (Biểu cảm cao)")
        raw_tts = input("Chọn (1-3) [mặc định: .env]: ").strip().lower()
        tts_engine = _TTS_ENGINE_MAP.get(raw_tts, "")

        print("\nVoice Pitch Method:")
        print("  1. none      — Giữ nguyên giọng TTS gốc [nhận Enter]")
        print("  2. shift     — Chỉnh tông trầm/bổng (Pitch Shift)")
        print("  3. clone     — Nhại ngữ điệu giọng gốc (F0 Cloning)")
        print("  4. hoat_ngon — Giọng tươi trẻ (+3.5 semitones, Cô gái hoạt ngôn)")
        raw_pitch = input("Chọn (1-4) [mặc định: 1]: ").strip().lower()
        pitch = _PITCH_METHOD_MAP.get(raw_pitch, "none")

        subs = input("\nXuất kèm Phụ đề Vietsub vào Video? (y/n) [mặc định: y]: ").strip().lower()
        subs_flag = "" if subs != 'n' else "--no-subs"

        summarize = input("Tạo Bản tóm tắt & Gợi ý tiêu đề? (y/n) [mặc định: y]: ").strip().lower()
        summarize_flag = "" if summarize != 'n' else "--skip-summarize"
    else:
        # Quick Auto Mode - Best Defaults (fully automated, no interactive pauses)
        print("  → ⚡ Đã chọn Chế độ 1-Click Quick Auto! Đang tự động thiết lập thông số chuẩn...")
        lang = "auto"
        transcribe_device = WHISPER_DEVICE
        trans_style = TRANSLATION_STYLE
        pitch = PITCH_METHOD
        tts_engine = ""
        interactive_flag = ""  # No -i flag: 1-Click runs fully automated without pausing
        no_ocr_flag = ""
        manual_trans_flag = ""
        subs_flag = ""
        summarize_flag = ""

    # Construct CLI command
    cmd = [
        get_python_executable(),
        "-u",  # Force unbuffered stdout/stderr in subprocess
        os.path.join(os.path.dirname(__file__), "cli_pipeline.py"),
        url,
        "--lang", lang,
        "--translation-style", trans_style,
        "--pitch", pitch,
        "--transcribe-device", transcribe_device,
    ]
    if tts_engine:
        cmd.extend(["--tts-engine", tts_engine])
    if interactive_flag:
        cmd.append(interactive_flag)
    if no_ocr_flag:
        cmd.append(no_ocr_flag)
    if manual_trans_flag:
        cmd.append(manual_trans_flag)
    if subs_flag:
        cmd.append(subs_flag)
    if summarize_flag:
        cmd.append(summarize_flag)

    print("\n🚀 Đang khởi chạy Pipeline...")
    print(f"Lệnh thực thi: {' '.join(cmd)}")
    print("="*55 + "\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline dừng lại với mã lỗi {e.returncode}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Tiến trình đã bị hủy bởi người dùng.")

    input("\nẤn Enter để quay lại menu chính...")

def list_and_resume_jobs():
    while True:
        jobs = scan_existing_jobs()
        print("\n" + "="*60)
        print("           📂 Quản Lý & Tiếp Tục Các Job Đã Chạy            ")
        print("="*60)
        
        if not jobs:
            print("  Chưa có job nào trong thư mục output/.")
            print("="*60)
            input("\nẤn Enter để quay lại menu chính...")
            return
            
        print(f"  {'#':<3} | {'Job ID':<10} | {'Trạng Thái':<18} | {'Video URL'}")
        print("-" * 60)
        for idx, job in enumerate(jobs, start=1):
            url_display = job["url"]
            if len(url_display) > 30:
                url_display = url_display[:27] + "..."
            print(f"  {idx:<3} | {job['job_id']:<10} | {job['status']:<18} | {url_display}")
        print("-" * 60)
        print("  0. Quay lại Menu chính")
        print("="*60)
        
        choice = input("Chọn Job cần xem/tiếp tục (nhập 0 để hủy): ").strip()
        if choice == '0' or not choice:
            return
            
        try:
            selection_idx = int(choice) - 1
            if selection_idx < 0 or selection_idx >= len(jobs):
                print("❌ Lựa chọn không hợp lệ.")
                continue
        except ValueError:
            print("❌ Vui lòng nhập số.")
            continue
            
        selected_job = jobs[selection_idx]
        job_id = selected_job["job_id"]
        
        while True:
            job_dir = os.path.join("output", job_id)
            summary_path = os.path.join(job_dir, "summary.txt")
            has_summary = os.path.exists(summary_path)

            print("\n" + "="*55)
            print(f"  Job ID : {job_id}")
            print(f"  URL    : {selected_job['url']}")
            print(f"  Status : {selected_job['status']}")
            print("="*55)
            print("  1. ⚡ Tiếp tục tự động từ bước chưa hoàn thành [Enter]")
            print("  2. 🎛️ Tiếp tục + Bật xem & duyệt checkpoint (--interactive)")
            print("  3. 🎯 Ép chạy lại từ một bước cụ thể...")
            print("  4. 🗑️ Xóa toàn bộ Job này (Folder & Checkpoints)")
            if has_summary:
                print("  5. 📄 Xem Bản Tóm Tắt & Gợi Ý Tiêu Đề (summary.txt)")
                print("  6. ⬅️ Quay lại")
            else:
                print("  5. ⬅️ Quay lại")
            print("="*55)
            
            action = input("Chọn thao tác [mặc định: 1]: ").strip() or "1"
            
            if (has_summary and action == '6') or (not has_summary and action == '5'):
                break
            
            if has_summary and action == '5':
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        print("\n" + f.read())
                except Exception as e:
                    print(f"❌ Lỗi đọc file summary: {e}")
                input("\nẤn Enter để quay lại...")
                continue
                
            cmd = [
                get_python_executable(),
                "-u",  # Force unbuffered stdout/stderr in subprocess
                os.path.join(os.path.dirname(__file__), "cli_pipeline.py"),
                "--resume", job_id
            ]

            if action == '1':
                pass  # Fully automated — no -i flag
            elif action == '2':
                cmd.append("-i")  # User explicitly wants interactive checkpoint review
            elif action == '3':
                print("\nCác bước trong Pipeline:")
                for step in STEPS:
                    print(f"  - {step}")
                resume_step = input("\nNhập tên bước muốn bắt đầu lại: ").strip().lower()
                if resume_step not in STEPS:
                    print("❌ Tên bước không hợp lệ.")
                    continue
                cmd.extend(["--resume-from", resume_step])
                
                use_interactive = input("Bật xem & duyệt checkpoint? (y/n) [mặc định: n]: ").strip().lower()
                if use_interactive == 'y':
                    cmd.append("-i")
                
                use_cache = input(f"Dùng lại dữ liệu cũ của '{resume_step}'? (y/n) [mặc định: y]: ").strip().lower()
                if use_cache == 'n':
                    cmd.append("--clear-cache")
            elif action == '4':
                confirm = input(f"⚠️ Bạn có chắc chắn muốn XÓA Job {job_id}? (y/n): ").strip().lower()
                if confirm == 'y':
                    job_dir = os.path.join("output", job_id)
                    if os.path.exists(job_dir):
                        shutil.rmtree(job_dir)
                    for f in glob.glob(os.path.join("output", f"{job_id}.*")):
                        try: os.remove(f)
                        except Exception: pass
                    print(f"✅ Đã xóa Job {job_id}.")
                    break
                else:
                    continue
            else:
                print("❌ Lựa chọn không hợp lệ.")
                continue

            print("\n🚀 Khởi chạy Pipeline...")
            print(f"Lệnh thực thi: {' '.join(cmd)}")
            print("="*55 + "\n")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Pipeline dừng lại với mã lỗi {e.returncode}")
            except KeyboardInterrupt:
                print("\n\n⚠️ Tiến trình đã bị hủy bởi người dùng.")

            input("\nẤn Enter để tiếp tục...")
            break

def main():
    while True:
        print("\n" + "="*55)
        print("          🎙️ AutoDub VN - Interactive TUI Menu        ")
        print("=======================================================")
        print("  1. 🚀 Khởi tạo Dubbing Job Mới (1-Click hoặc Tùy chỉnh)")
        print("  2. 📂 Quản lý & Tiếp tục các Job Đã Chạy (Checkpoints)")
        print("  3. ⚙️ Xem Cấu Hình Hệ Thống (Configuration)")
        print("  4. ❌ Thoát")
        print("="*55)
        
        choice = input("Nhập lựa chọn (1-4): ").strip()
        
        if choice == '1':
            start_new_job()
        elif choice == '2':
            list_and_resume_jobs()
        elif choice == '3':
            show_config()
        elif choice == '4':
            print("\nTam biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng chọn 1-4.")
            input("Ấn Enter để thử lại...")

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1].startswith("http://") or sys.argv[1].startswith("https://") or os.path.exists(sys.argv[1])):
        start_new_job(initial_url=sys.argv[1])
    else:
        main()
