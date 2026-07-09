# TÀI LIỆU ĐẶC TẢ SẢN PHẨM
## Ứng dụng Dịch & Lồng tiếng Video Tự động
*(Tên tạm gọi: VietDub / AutoDub VN)*

**Phiên bản:** 1.0
**Ngày cập nhật:** 07/07/2026

---

## 1. Tổng quan sản phẩm

### 1.1. Mục tiêu

Xây dựng một ứng dụng web cho phép người dùng dán đường link video, hệ thống tự động dịch nội dung từ ngôn ngữ gốc sang tiếng Việt, lồng tiếng bằng giọng đọc tổng hợp (TTS), đồng thời chèn phụ đề và logo chìm (watermark) lên video, trả về một video hoàn chỉnh để người dùng tải về hoặc xem trực tiếp.

### 1.2. Phạm vi ngôn ngữ (giai đoạn đầu)

| Chiều | Ngôn ngữ hỗ trợ |
|---|---|
| Ngôn ngữ đầu vào | Tiếng Anh, Tiếng Trung |
| Ngôn ngữ đầu ra | Tiếng Việt |

### 1.3. Đối tượng người dùng

- Người làm nội dung (content creator) muốn Việt hóa nhanh video nước ngoài
- Cá nhân/tổ chức cần xem hiểu nội dung video tiếng Anh, tiếng Trung bằng tiếng Việt
- Nhà sáng tạo cần gắn thương hiệu (watermark) lên video đã Việt hóa

---

## 2. Luồng xử lý nghiệp vụ (Business Flow)

### 2.1. Các bước xử lý chính

1. Người dùng dán link video (YouTube, TikTok, Facebook, v.v.)
2. Hệ thống tải video và trích xuất audio
3. Nhận diện ngôn ngữ gốc (Anh/Trung) — tự động hoặc do người dùng chọn
4. Speech-to-Text: chuyển giọng nói gốc thành văn bản
5. Dịch máy: dịch văn bản sang tiếng Việt
6. Text-to-Speech: tạo giọng đọc tiếng Việt từ bản dịch
7. Đồng bộ thời lượng giọng đọc mới với video gốc
8. Ghép audio lồng tiếng mới vào video
9. Chèn logo chìm (watermark) lên video
10. Ghi phụ đề tiếng Việt lên video (hardsub)
11. Xuất video hoàn chỉnh, cho phép xem trước, tải về hoặc chia sẻ

### 2.2. Luồng sử dụng của người dùng (User Flow)

- **Bước 1:** Dán link video vào ô nhập liệu
- **Bước 2:** Chọn ngôn ngữ gốc (Anh/Trung) hoặc để hệ thống tự nhận diện
- **Bước 3:** Nhấn nút "Dịch & Lồng tiếng"
- **Bước 4:** Theo dõi tiến trình xử lý theo thời gian thực (tải video → nhận diện giọng nói → dịch → tạo giọng lồng tiếng → ghép video)
- **Bước 5:** Vào màn hình "Xem trước & Tùy chỉnh" để chỉnh sửa phụ đề, bật/tắt watermark, chỉnh vị trí/màu sắc sub
- **Bước 6:** Nhấn "Xuất video" để render bản cuối cùng
- **Bước 7:** Tải video về hoặc lấy link chia sẻ

---

## 3. Đặc tả tính năng chi tiết

### 3.1. Dịch & Lồng tiếng

**Mô tả**
Tự động chuyển đổi lời thoại trong video từ ngôn ngữ gốc (Anh/Trung) sang giọng đọc tiếng Việt, giữ nguyên hình ảnh gốc của video.

**Các bước kỹ thuật**
- Trích xuất audio từ video (yt-dlp hoặc công cụ tương đương)
- Speech-to-Text bằng mô hình nhận diện giọng nói đa ngôn ngữ (ví dụ Whisper)
- Dịch văn bản sang tiếng Việt bằng API dịch hoặc mô hình ngôn ngữ lớn (LLM), ưu tiên bản dịch tự nhiên, đúng ngữ cảnh
- Text-to-Speech tiếng Việt tạo giọng đọc tự nhiên, có ngữ điệu
- Căn chỉnh thời lượng câu đọc mới khớp với đoạn thoại gốc (co giãn tốc độ đọc hoặc rút gọn câu dịch nếu cần)
- Ghép track audio mới vào video bằng FFmpeg, giữ nguyên hình ảnh

**Lưu ý / ràng buộc**
- Câu tiếng Việt dịch ra thường dài hơn bản gốc tiếng Anh/Trung → cần xử lý co giãn tốc độ hoặc rút gọn để khớp thời lượng
- Video có nhiều người nói (multi-speaker) cần tách giọng để lồng tiếng riêng biệt cho từng nhân vật (tính năng nâng cao)

### 3.2. Chèn logo chìm (Watermark)

**Mô tả**
Cho phép người dùng chèn một logo mờ (watermark) lên video đầu ra để đánh dấu bản quyền/thương hiệu.

**Yêu cầu chức năng**
- Người dùng tải lên ảnh logo (khuyến khích định dạng PNG nền trong suốt)
- Tùy chỉnh vị trí: 4 góc màn hình hoặc chính giữa
- Tùy chỉnh kích thước logo và độ trong suốt (opacity)
- Tùy chỉnh khoảng cách lề (margin) so với mép khung hình
- Lưu logo mặc định trong hồ sơ người dùng để tái sử dụng cho các lần xử lý sau

**Kỹ thuật thực hiện**
Dùng FFmpeg filter `overlay` để chèn logo lên từng khung hình video mà không làm giảm chất lượng hình ảnh gốc.

### 3.3. Ghi phụ đề lên video (Hardsub)

**Mô tả**
Tự động sinh phụ đề tiếng Việt từ bản dịch kèm mốc thời gian (timestamp), cho phép người dùng xem trước, chỉnh sửa, và ghi cứng (burn) phụ đề vào khung hình video.

**Yêu cầu chức năng**
- Tự động tạo file phụ đề định dạng `.srt` hoặc `.ass` từ bản dịch
- Cho phép người dùng xem trước và chỉnh sửa nội dung, thời gian hiển thị của từng dòng sub
- Tùy chỉnh giao diện hiển thị: font chữ, cỡ chữ, màu chữ, viền chữ, vị trí (trên/dưới khung hình), nền mờ phía sau chữ
- Cho phép xuất kèm file phụ đề rời (`.srt`) song song với video, phòng trường hợp người dùng muốn dùng sub mềm thay vì burn cứng

**Kỹ thuật thực hiện**
Dùng FFmpeg với filter `subtitles` (thư viện `libass`) để burn phụ đề trực tiếp vào video.

### 3.4. Màn hình Xem trước & Tùy chỉnh

Trước khi xuất video cuối cùng, người dùng được đưa tới màn hình cho phép bật/tắt và tinh chỉnh từng thành phần:

- Bật/tắt lồng tiếng
- Bật/tắt watermark — chọn ảnh, vị trí, độ mờ
- Bật/tắt phụ đề — chỉnh sửa nội dung, font, màu sắc, vị trí
- Nút "Xem trước nhanh" (preview 10–15 giây) trước khi render toàn bộ video, giúp tiết kiệm thời gian và chi phí xử lý

---

## 4. Kiến trúc kỹ thuật đề xuất

### 4.1. Các thành phần hệ thống

| Thành phần | Công nghệ đề xuất | Vai trò |
|---|---|---|
| Trích xuất video/audio | yt-dlp | Tải video và tách audio từ đường link |
| Speech-to-Text | Whisper (OpenAI) hoặc tương đương | Chuyển giọng nói gốc (Anh/Trung) thành văn bản |
| Dịch máy | Google Translate / DeepL / LLM | Dịch văn bản sang tiếng Việt tự nhiên, đúng ngữ cảnh |
| Text-to-Speech | Google TTS / Azure TTS / TTS tiếng Việt chuyên biệt | Tạo giọng đọc tiếng Việt tự nhiên |
| Xử lý video | FFmpeg | Ghép audio, chèn watermark, burn phụ đề |

### 4.2. Thứ tự xử lý FFmpeg

Để hạn chế giảm chất lượng do nén lại nhiều lần, các bước chỉnh sửa hình ảnh nên được gộp trong một lệnh xử lý duy nhất:

```
Video gốc → Thay audio (giọng lồng tiếng) → Overlay logo (watermark) → Burn phụ đề (libass) → Xuất video output
```

### 4.3. Tiến trình xử lý hiển thị cho người dùng

- Đang tải video
- Đang nhận diện giọng nói
- Đang dịch nội dung
- Đang tạo giọng lồng tiếng
- Đang ghép video (audio + watermark + phụ đề)

---

## 5. Rủi ro & Thách thức kỹ thuật

| Thách thức | Mô tả / Hướng xử lý |
|---|---|
| Đồng bộ thời gian | Câu tiếng Việt thường dài hơn bản gốc → cần co giãn tốc độ đọc hoặc rút gọn câu dịch |
| Bản quyền video | Tải về và chỉnh sửa video có bản quyền có thể vi phạm điều khoản nền tảng gốc |
| Chi phí xử lý | Speech-to-Text, dịch, TTS đều tốn chi phí API, đặc biệt với video dài |
| Chất lượng giọng đọc | Giọng TTS tiếng Việt cần tự nhiên, tránh nghe như robot |
| Nhiều người nói | Cần tách giọng (speaker diarization) để lồng tiếng riêng cho từng nhân vật |
| Thời gian render | Burn sub + watermark yêu cầu encode lại toàn bộ video, tăng thời gian xử lý |
| Bố cục hiển thị | Cần tự động điều chỉnh vị trí logo/sub phù hợp với tỷ lệ khung hình dọc/ngang khác nhau |

---

## 6. Tính năng mở rộng (đề xuất cho các giai đoạn sau)

- Chọn giọng đọc (nam/nữ, giọng vùng miền Bắc/Nam)
- Hỗ trợ thêm ngôn ngữ đầu vào khác (Hàn, Nhật...)
- Lưu lịch sử các video đã xử lý
- Tách giọng nhiều nhân vật (multi-speaker dubbing)
- Xuất phụ đề song ngữ

---

## 7. Tóm tắt phạm vi phiên bản 1.0 (MVP)

| Hạng mục | Có trong MVP? |
|---|---|
| Dịch & lồng tiếng (Anh/Trung → Việt) | Có |
| Chèn logo chìm (watermark) | Có |
| Ghi phụ đề lên video (hardsub) | Có |
| Xuất phụ đề rời (.srt) | Có |
| Chỉnh sửa bản dịch/sub trước khi xuất | Có |
| Tách giọng nhiều người nói | Không (giai đoạn sau) |
| Chọn giọng đọc theo vùng miền | Không (giai đoạn sau) |
| Hỗ trợ thêm ngôn ngữ đầu vào (Hàn, Nhật...) | Không (giai đoạn sau) |
