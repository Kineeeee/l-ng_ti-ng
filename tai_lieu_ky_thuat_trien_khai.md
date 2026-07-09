# TÀI LIỆU KỸ THUẬT TRIỂN KHAI
## VietDub / AutoDub VN — Implementation Guide cho AI Agent

**Phiên bản:** 1.0
**Mục đích:** Tài liệu này được viết để đưa cho một AI coding agent (Claude Code, Cursor, v.v.) đọc và triển khai trực tiếp dự án. Mọi quyết định kiến trúc, thứ tự thực hiện, cấu trúc thư mục, API contract, và pipeline xử lý đều được đặc tả cụ thể để AI có thể bắt tay vào code ngay mà không cần hỏi lại thêm về mặt kiến trúc.

**Nguồn gốc:** Tài liệu này là bước triển khai kỹ thuật tiếp theo của `dac_ta_ung_dung.md` (tài liệu đặc tả sản phẩm). Đọc file đó trước để hiểu bối cảnh nghiệp vụ nếu cần.

---

## 0. Nguyên tắc triển khai tổng quát

1. **Thứ tự bắt buộc:** Kiến trúc & tech stack → Pipeline xử lý lõi (CLI trước, chạy được trên 1 video mẫu) → Bọc thành API → Dựng UI. KHÔNG viết UI trước khi pipeline CLI chạy ổn định.
2. **Pipeline phải chạy độc lập được qua CLI/script trước khi bọc vào FastAPI.** Mục tiêu: xác nhận từng bước (tải video, STT, dịch, TTS, ghép FFmpeg) hoạt động đúng trên ít nhất 1 video mẫu tiếng Anh và 1 video mẫu tiếng Trung, trước khi viết bất kỳ dòng code API/UI nào.
3. **Mỗi bước xử lý là một module Python độc lập, có thể test riêng lẻ** (unit-testable), không viết thành một script monolithic duy nhất.
4. **Xử lý video là tác vụ nặng và lâu (có thể vài phút)** → bắt buộc dùng job queue (Celery/RQ), không xử lý đồng bộ trong HTTP request.
5. **Toàn bộ trạng thái xử lý phải được lưu và truy vấn được qua job_id**, để frontend polling tiến trình.

---

## 1. Kiến trúc & Tech Stack

### 1.1. Sơ đồ luồng dữ liệu tổng thể

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Frontend  │─────▶│  FastAPI     │─────▶│  Redis (Queue)  │
│  (Next.js)  │◀─────│  (REST API)  │      │                 │
└─────────────┘ poll └──────┬───────┘      └────────┬────────┘
       │                    │                        │
       │              ┌─────▼──────┐          ┌──────▼───────┐
       │              │  Postgres  │          │ Celery Worker│
       │              │ (job meta) │          │ (xử lý video)│
       │              └────────────┘          └──────┬───────┘
       │                                              │
       │                                       ┌──────▼───────┐
       └──────────────────────────────────────▶│ Object Storage│
                  (tải video kết quả)           │ (S3/local)    │
                                                 └───────────────┘
```

### 1.2. Tech stack cụ thể

| Lớp | Công nghệ | Phiên bản đề xuất | Ghi chú |
|---|---|---|---|
| Frontend | Next.js + TypeScript | 14.x (App Router) | UI, polling job status |
| Styling | Tailwind CSS | 3.x | |
| Backend API | Python + FastAPI | 3.11 / 0.11x | Async endpoints |
| Job Queue | Celery | 5.x | Worker xử lý pipeline nặng |
| Message Broker | Redis | 7.x | Broker cho Celery + cache job status |
| Metadata DB | PostgreSQL | 15.x | Lưu thông tin job, user, lịch sử |
| Video download | yt-dlp | latest | Tải video từ link |
| Speech-to-Text | faster-whisper (hoặc openai-whisper) | large-v3 / medium | Chạy local hoặc qua API OpenAI |
| Dịch máy | LLM API (Claude/GPT) hoặc DeepL API | — | Ưu tiên LLM để dịch tự nhiên theo ngữ cảnh |
| Text-to-Speech | Azure TTS (giọng vi-VN) hoặc FPT.AI TTS / Google Cloud TTS | — | Cần giọng tiếng Việt tự nhiên |
| Xử lý video/audio | FFmpeg (subprocess qua `ffmpeg-python` hoặc gọi CLI trực tiếp) | 6.x | Ghép audio, overlay logo, burn sub |
| Storage | AWS S3 / MinIO (self-host) / local disk (dev) | — | Lưu file video gốc, trung gian, output |
| Containerize | Docker + docker-compose | — | Đóng gói toàn bộ stack (api, worker, redis, postgres) |

### 1.3. Cấu trúc thư mục dự án (monorepo)

```
vietdub/
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py                 # FastAPI entrypoint
│   │   ├── config.py               # Đọc biến môi trường
│   │   ├── api/
│   │   │   ├── routes_jobs.py      # POST /jobs, GET /jobs/{id}
│   │   │   └── routes_assets.py    # Upload logo, download video output
│   │   ├── models/
│   │   │   ├── job.py              # SQLAlchemy model Job
│   │   │   └── schemas.py          # Pydantic request/response schemas
│   │   ├── db/
│   │   │   └── session.py
│   │   ├── worker/
│   │   │   ├── celery_app.py
│   │   │   └── tasks.py            # Task pipeline chính (orchestrator)
│   │   ├── pipeline/
│   │   │   ├── download.py         # Module 1: yt-dlp download
│   │   │   ├── transcribe.py       # Module 2: Speech-to-Text (Whisper)
│   │   │   ├── translate.py        # Module 3: Dịch sang tiếng Việt
│   │   │   ├── tts.py              # Module 4: Text-to-Speech tiếng Việt
│   │   │   ├── align.py            # Module 5: Đồng bộ thời lượng
│   │   │   ├── render.py           # Module 6: FFmpeg ghép audio+watermark+sub
│   │   │   └── subtitle.py         # Sinh file .srt/.ass từ bản dịch
│   │   └── storage/
│   │       └── s3_client.py        # Wrapper upload/download storage
│   └── tests/
│       ├── test_download.py
│       ├── test_transcribe.py
│       ├── test_translate.py
│       ├── test_tts.py
│       └── test_render.py
├── frontend/
│   ├── package.json
│   ├── app/
│   │   ├── page.tsx                 # Màn hình "Dán link"
│   │   ├── jobs/[id]/page.tsx       # Màn hình tiến trình + Xem trước & Tùy chỉnh
│   │   └── components/
│   │       ├── LinkInputForm.tsx
│   │       ├── ProgressTracker.tsx
│   │       ├── PreviewCustomizePanel.tsx
│   │       ├── WatermarkSettings.tsx
│   │       └── SubtitleEditor.tsx
│   └── lib/
│       └── api.ts                   # Client gọi FastAPI
└── scripts/
    └── cli_pipeline.py               # Script CLI test toàn bộ pipeline độc lập (BƯỚC ĐẦU TIÊN CẦN VIẾT)
```

---

## 2. API Contract

### 2.1. Job lifecycle (trạng thái)

```
PENDING → DOWNLOADING → TRANSCRIBING → TRANSLATING → SYNTHESIZING_TTS
        → ALIGNING → RENDERING → READY_FOR_PREVIEW → FINALIZING → DONE
        (mỗi bước có thể chuyển sang FAILED kèm error_message)
```

### 2.2. Endpoints

**`POST /jobs`** — Tạo job xử lý mới
```json
// Request
{
  "video_url": "https://youtube.com/watch?v=...",
  "source_language": "auto | en | zh",
  "options": {
    "enable_dubbing": true,
    "enable_watermark": false,
    "enable_subtitle": true
  }
}
// Response (201)
{
  "job_id": "job_abc123",
  "status": "PENDING",
  "created_at": "2026-07-07T10:00:00Z"
}
```

**`GET /jobs/{job_id}`** — Lấy trạng thái & kết quả job
```json
{
  "job_id": "job_abc123",
  "status": "TRANSLATING",
  "progress_percent": 45,
  "current_step_label": "Đang dịch nội dung",
  "detected_language": "en",
  "error_message": null,
  "preview_video_url": null,
  "final_video_url": null,
  "subtitle_segments": [
    {"id": 1, "start": 0.0, "end": 2.4, "original_text": "Hello everyone", "translated_text": "Xin chào mọi người"}
  ]
}
```

**`PATCH /jobs/{job_id}/subtitles`** — Chỉnh sửa phụ đề trước khi render cuối
```json
// Request
{
  "subtitle_segments": [
    {"id": 1, "start": 0.0, "end": 2.4, "translated_text": "Xin chào tất cả mọi người"}
  ]
}
```

**`POST /jobs/{job_id}/customize`** — Cập nhật tùy chọn watermark/sub trước khi xuất bản cuối
```json
{
  "watermark": {"enabled": true, "image_url": "s3://.../logo.png", "position": "bottom-right", "opacity": 0.6, "scale": 0.15},
  "subtitle_style": {"font": "Arial", "size": 24, "color": "#FFFFFF", "outline_color": "#000000", "position": "bottom"}
}
```

**`POST /jobs/{job_id}/render-final`** — Trigger render bản cuối (sau khi user xem preview & chỉnh xong)

**`POST /assets/logo`** — Upload logo watermark (multipart/form-data) → trả về `image_url`

**`GET /jobs/{job_id}/download`** — Redirect / trả link tải video cuối (và `.srt` nếu có)

---

## 3. Đặc tả chi tiết Pipeline xử lý (theo module)

> Nguyên tắc: mỗi module nhận input rõ ràng, trả output rõ ràng, có thể test độc lập bằng CLI trước khi nối vào Celery task.

### Module 1 — `download.py`: Tải video

- **Input:** `video_url: str`
- **Output:** `{ "video_path": str, "audio_path": str, "duration_sec": float, "resolution": (w,h) }`
- **Công cụ:** `yt-dlp` (gọi qua Python API `yt_dlp.YoutubeDL`, không qua subprocess để bắt lỗi tốt hơn)
- **Xử lý:** Tải video ở chất lượng vừa đủ (720p đề xuất để giảm thời gian render), tách audio riêng bằng `ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav` (16kHz mono, chuẩn input cho Whisper)
- **Lỗi cần xử lý:** link không hợp lệ, video riêng tư/bị khóa, video quá dài (đặt giới hạn ví dụ 30 phút cho MVP)

### Module 2 — `transcribe.py`: Speech-to-Text

- **Input:** `audio_path: str`, `language_hint: Optional[str]` ("en" | "zh" | None để tự nhận diện)
- **Output:** `List[Segment]` với `{ id, start, end, text, detected_language }`
- **Công cụ:** `faster-whisper` (model `medium` hoặc `large-v3`, tự động detect ngôn ngữ nếu `language_hint=None`)
- **Chạy local (GPU nếu có) hoặc gọi OpenAI Whisper API** — đề xuất bắt đầu bằng API để đơn giản hoá MVP, chuyển sang self-host nếu chi phí cao
- **Output segment giữ nguyên timestamp gốc** — đây là input bắt buộc cho bước align sau này

### Module 3 — `translate.py`: Dịch sang tiếng Việt

- **Input:** `List[Segment]` (từ module 2), `source_language: str`
- **Output:** `List[Segment]` thêm field `translated_text`
- **Công cụ:** Gọi LLM (Claude/GPT) theo batch, KHÔNG dịch từng câu riêng lẻ — gửi cả đoạn hội thoại kèm ngữ cảnh (câu trước/sau) để bản dịch tự nhiên hơn
- **Prompt gợi ý cho LLM:**
  ```
  Bạn là biên dịch viên chuyên nghiệp. Dịch đoạn hội thoại sau từ {source_language} sang
  tiếng Việt tự nhiên, đúng ngữ cảnh, giữ văn phong phù hợp với nội dung video.
  Trả về đúng định dạng JSON list, giữ nguyên số lượng segment, KHÔNG gộp/tách câu.
  Input: [{"id": 1, "text": "..."}, ...]
  ```
- **Ràng buộc quan trọng:** số lượng segment đầu ra PHẢI khớp đầu vào (để giữ mapping timestamp)

### Module 4 — `tts.py`: Text-to-Speech tiếng Việt

- **Input:** `List[Segment]` (đã có `translated_text`), `voice_id: str` (giọng đọc)
- **Output:** `List[Segment]` thêm field `tts_audio_path`, `tts_duration`
- **Công cụ:** Azure TTS (giọng `vi-VN-HoaiMyNeural` hoặc `vi-VN-NamMinhNeural`) — hỗ trợ điều chỉnh tốc độ đọc (`rate`) qua SSML
- **Sinh audio riêng cho từng segment** (không gộp), để dễ căn chỉnh timing ở bước sau

### Module 5 — `align.py`: Đồng bộ thời lượng

- **Input:** `List[Segment]` (có `start`, `end` gốc và `tts_duration`)
- **Output:** `List[Segment]` thêm `adjusted_start`, `adjusted_speed_ratio`
- **Logic xử lý:**
  1. Với mỗi segment, so sánh `tts_duration` với `(end - start)` gốc
  2. Nếu `tts_duration > (end - start) * 1.15` → tăng tốc độ đọc TTS (dùng SSML `rate` hoặc `ffmpeg atempo`) để khớp, giới hạn tối đa tăng tốc 1.3x (tránh nghe kỳ dị)
  3. Nếu vẫn không khớp sau khi tăng tốc tối đa → cho phép segment audio kéo dài lấn sang khoảng lặng của segment kế tiếp (nếu có), hoặc chấp nhận lệch nhỏ
  4. Trả về timeline audio cuối cùng để ghép

### Module 6 — `subtitle.py`: Sinh file phụ đề

- **Input:** `List[Segment]` (có `translated_text`, `start`, `end`)
- **Output:** file `.srt` và `.ass`
- **Định dạng `.srt` chuẩn:**
  ```
  1
  00:00:00,000 --> 00:00:02,400
  Xin chào mọi người
  ```
- **`.ass`** dùng khi cần style tùy chỉnh (font, màu, viền) để burn bằng `libass`

### Module 7 — `render.py`: Ghép video cuối (FFmpeg)

- **Input:** `video_path`, danh sách audio segment đã align, `watermark_config`, `subtitle_ass_path`, `subtitle_enabled`, `watermark_enabled`
- **Output:** `final_video_path`
- **Lệnh FFmpeg gộp (ví dụ, 1 lần encode duy nhất):**
  ```bash
  ffmpeg -i input_video.mp4 -i dubbed_audio.wav -i logo.png \
    -filter_complex "[0:v][2:v] overlay=W-w-20:H-h-20:enable='between(t,0,999999)' [wm]; \
                     [wm] subtitles=subs.ass [outv]" \
    -map "[outv]" -map 1:a \
    -c:v libx264 -crf 20 -preset medium -c:a aac \
    output_final.mp4
  ```
  - Nếu `watermark_enabled=false` → bỏ nhánh overlay logo
  - Nếu `subtitle_enabled=false` → bỏ filter `subtitles=`
  - `dubbed_audio.wav` là audio đã ghép nối tất cả segment TTS theo `adjusted_start` (bước riêng trước khi vào FFmpeg, dùng `pydub` hoặc `ffmpeg concat`)

---

## 4. Kế hoạch triển khai theo giai đoạn (Task Breakdown cho AI Agent)

### Giai đoạn A — Pipeline CLI độc lập (làm trước tiên, không đụng vào API/UI)

- [ ] A1. Viết `scripts/cli_pipeline.py`: nhận `video_url` qua argument, chạy tuần tự Module 1 → 7, in log từng bước, xuất video kết quả ra thư mục `output/`
- [ ] A2. Test với 1 video mẫu tiếng Anh (~1-2 phút) → xác nhận: audio lồng tiếng nghe tự nhiên, timing không lệch quá 0.5s
- [ ] A3. Test với 1 video mẫu tiếng Trung → xác nhận STT nhận diện tiếng Trung chính xác
- [ ] A4. Test watermark: chèn 1 logo PNG mẫu, xác nhận vị trí/opacity đúng
- [ ] A5. Test hardsub: xác nhận phụ đề hiển thị đúng timing, không đè lên watermark

**Chỉ chuyển sang Giai đoạn B khi Giai đoạn A chạy ổn định trên ít nhất 2 video mẫu (1 Anh, 1 Trung).**

### Giai đoạn B — Backend API + Job Queue

- [ ] B1. Setup FastAPI project, PostgreSQL model `Job`, migration (Alembic)
- [ ] B2. Setup Celery + Redis, chuyển pipeline CLI (giai đoạn A) thành Celery task `process_video_job(job_id)`
- [ ] B3. Implement các endpoint theo mục 2.2 (`POST /jobs`, `GET /jobs/{id}`, `PATCH /jobs/{id}/subtitles`, `POST /jobs/{id}/customize`, `POST /jobs/{id}/render-final`)
- [ ] B4. Setup storage (S3/MinIO), upload video output + trả signed URL
- [ ] B5. Viết test tích hợp: gọi `POST /jobs` → poll `GET /jobs/{id}` đến khi `DONE`

### Giai đoạn C — Frontend

- [ ] C1. Trang chủ: form dán link + chọn ngôn ngữ gốc → gọi `POST /jobs`
- [ ] C2. Trang job detail: polling `GET /jobs/{id}` mỗi 2s, hiển thị `ProgressTracker` theo `current_step_label`
- [ ] C3. Khi status = `READY_FOR_PREVIEW`: hiển thị `PreviewCustomizePanel` (bật/tắt watermark, sub, chỉnh sửa từng dòng phụ đề qua `SubtitleEditor`)
- [ ] C4. Nút "Xuất video" gọi `POST /jobs/{id}/render-final`, tiếp tục polling đến `DONE`
- [ ] C5. Khi `DONE`: hiển thị nút tải video + tải `.srt`

### Giai đoạn D — Đóng gói & vận hành

- [ ] D1. Viết `docker-compose.yml` (api, worker, redis, postgres, frontend)
- [ ] D2. Biến môi trường: API key (Whisper/OpenAI, Azure TTS, LLM dịch), S3 credentials
- [ ] D3. Giới hạn resource: max video length, rate limit số job đồng thời/user (tránh quá tải worker)

---

## 5. Ràng buộc & Lưu ý quan trọng khi AI triển khai

- **Không tự ý đổi thứ tự pipeline** (download → transcribe → translate → tts → align → render). Thứ tự này đảm bảo timestamp gốc được giữ xuyên suốt để align chính xác.
- **Giữ mapping segment ID xuyên suốt các module** — đây là điểm dễ gây lỗi nhất nếu số lượng segment bị thay đổi giữa các bước (đặc biệt ở bước dịch LLM).
- **Video dài giới hạn 30 phút cho MVP** để tránh timeout worker và chi phí API tăng vọt.
- **Luôn xử lý watermark và subtitle như 2 filter độc lập, có thể bật/tắt riêng** — không hardcode cả 2 luôn bật.
- **Không dùng lại `openai-whisper` (bản gốc chậm) cho production** — ưu tiên `faster-whisper` hoặc API để tối ưu tốc độ/chi phí.
- **File tạm (audio segment, video trung gian) cần dọn dẹp sau khi job hoàn tất** để tránh đầy ổ đĩa worker.
