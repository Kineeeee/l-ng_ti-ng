"""
pitch.py — Pitch Processing Module for TTS Dubbing Pipeline

Hai phương pháp xử lý cao độ (pitch):

  Method 1 — Simple Pitch Shift (nhanh, nhẹ):
    Phân tích pitch trung bình của giọng gốc trong mỗi segment,
    rồi shift pitch của TTS output theo delta semitones.
    Dùng librosa.effects.pitch_shift hoặc ffmpeg rubberband.

  Method 2 — F0 Contour Cloning (chất lượng cao):
    Extract đường cong F0 (fundamental frequency theo thời gian)
    từ audio gốc, warp nó lên trục thời gian của TTS, rồi
    resynthesis audio với đường cong F0 mới dùng WORLD vocoder
    (pyworld) hoặc Praat (parselmouth).

Cách dùng:
    from backend.app.pipeline.pitch import apply_pitch_to_segment

    # Method 1 — shift pitch đơn giản
    apply_pitch_to_segment(
        tts_wav_path="tts_1.wav",
        orig_wav_path="original.wav",
        orig_start_sec=1.2,
        orig_end_sec=3.5,
        output_path="tts_1_pitched.wav",
        method="shift"
    )

    # Method 2 — clone F0 contour
    apply_pitch_to_segment(
        tts_wav_path="tts_1.wav",
        orig_wav_path="original.wav",
        orig_start_sec=1.2,
        orig_end_sec=3.5,
        output_path="tts_1_pitched.wav",
        method="clone"
    )
"""

import os
import sys
import wave
import types
import tempfile
import subprocess
import numpy as np
from typing import Optional

# ─── Python 3.14 compatibility: pkg_resources shim ───────────────────────────
# pyworld dùng pkg_resources để đọc version, nhưng Python 3.14 đã bỏ module này.
# Inject shim để pyworld import không bị lỗi.
if "pkg_resources" not in sys.modules:
    _shim = types.ModuleType("pkg_resources")
    class _FakeDist:
        version = "0.0.0"
    _shim.get_distribution = lambda *a, **kw: _FakeDist()
    sys.modules["pkg_resources"] = _shim

# ─── Lazy imports (tránh crash nếu chưa cài) ─────────────────────────────────
def _import_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        raise ImportError("librosa chưa được cài. Chạy: pip install librosa soundfile")


def _import_pyworld():
    try:
        import pyworld as pw
        return pw
    except ImportError:
        raise ImportError("pyworld chưa được cài. Chạy: pip install pyworld")


def _import_parselmouth():
    try:
        import parselmouth
        return parselmouth
    except ImportError:
        raise ImportError("praat-parselmouth chưa được cài. Chạy: pip install praat-parselmouth")

# ─── Hằng số ──────────────────────────────────────────────────────────────────
ANALYSIS_SR = 22050      # Sample rate nội bộ để phân tích (librosa default)
PYWORLD_SR  = 16000      # pyworld hoạt động tốt nhất ở 16kHz hoặc 22050Hz
VOICED_THRESHOLD = 0.0   # Ngưỡng phân biệt voiced/unvoiced trong pyworld

# ─── Utility ──────────────────────────────────────────────────────────────────

def _load_mono_float32(wav_path: str, sr: int = ANALYSIS_SR) -> tuple[np.ndarray, int]:
    """Load WAV file thành numpy float32 mono array."""
    librosa = _import_librosa()
    y, actual_sr = librosa.load(wav_path, sr=sr, mono=True)
    return y, actual_sr


def _save_float32_to_wav(y: np.ndarray, sr: int, output_path: str):
    """Lưu numpy float32 array thành WAV file."""
    import soundfile as sf
    # Clip để tránh clipping
    y = np.clip(y, -1.0, 1.0)
    sf.write(output_path, y, sr, format="WAV", subtype="PCM_16")


def _hz_to_semitones(f_target: float, f_source: float) -> float:
    """Tính số semitones cần shift từ f_source lên f_target."""
    if f_source <= 0 or f_target <= 0:
        return 0.0
    return 12.0 * np.log2(f_target / f_source)


def _median_voiced_f0(f0: np.ndarray) -> float:
    """Tính trung vị F0 của các frame voiced (f0 > 0)."""
    voiced = f0[f0 > VOICED_THRESHOLD]
    if len(voiced) == 0:
        return 0.0
    return float(np.median(voiced))


# ─── Phương án 1: Simple Pitch Shift ─────────────────────────────────────────

def _extract_mean_pitch_librosa(wav_path: str, start_sec: float = 0.0, end_sec: float = None) -> float:
    """
    Dùng librosa.pyin để extract F0 trung bình của đoạn [start_sec, end_sec].
    Trả về Hz, hoặc 0.0 nếu không tìm được pitch.
    """
    librosa = _import_librosa()

    y, sr = _load_mono_float32(wav_path, sr=ANALYSIS_SR)

    # Cắt đoạn cần phân tích
    if start_sec > 0 or end_sec is not None:
        start_sample = int(start_sec * sr)
        end_sample   = int(end_sec * sr) if end_sec is not None else len(y)
        y = y[start_sample:end_sample]

    if len(y) < sr * 0.1:  # Đoạn quá ngắn (< 100ms)
        return 0.0

    # pyin — probabilistic YIN (tốt hơn yin thông thường)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),   # ~65Hz
        fmax=librosa.note_to_hz("C7"),   # ~2093Hz
        sr=sr,
        frame_length=2048,
    )

    voiced_f0 = f0[voiced_flag & (f0 > 0)]
    if len(voiced_f0) == 0:
        return 0.0

    return float(np.median(voiced_f0))


def _pitch_shift_clean(input_wav: str, output_wav: str, n_steps: float) -> bool:
    """
    Pitch shift âm thanh sạch, TRIỆT ĐỂ LOẠI BỎ HỆU ỨNG VANG (Phase Smearing / Reverb).

    Thứ tự ưu tiên:
    1. FFmpeg rubberband filter với pitchq=quality & formant=preserved
    2. FFmpeg asetrate + atempo (Resampling method — 100% không dính vang STFT)
    3. Librosa pitch_shift fallback
    """
    if abs(n_steps) < 0.1:
        import shutil
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return True

    pitch_ratio = 2.0 ** (n_steps / 12.0)

    # 1. FFmpeg rubberband filter (Thuật toán chuyên nghiệp — Giữ nguyên formant, không bị vang)
    cmd_rubberband = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", input_wav,
        "-af", f"rubberband=pitch={pitch_ratio:.6f}:pitchq=quality:formant=preserved",
        output_wav
    ]
    try:
        res = subprocess.run(cmd_rubberband, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 100:
            return True
    except Exception:
        pass

    # 2. Resampling method qua FFmpeg (asetrate + atempo — 100% Không vang phase vocoder)
    try:
        tempo_ratio = 1.0 / pitch_ratio
        sr = 24000
        try:
            with wave.open(input_wav, "rb") as wf:
                sr = wf.getframerate()
        except Exception:
            pass

        target_rate = int(sr * pitch_ratio)
        af_filter = f"asetrate={target_rate},aresample={sr},atempo={tempo_ratio:.6f}"
        cmd_resample = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", input_wav,
            "-af", af_filter,
            output_wav
        ]
        res = subprocess.run(cmd_resample, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 100:
            return True
    except Exception:
        pass

    # 3. Fallback: Librosa
    librosa = _import_librosa()
    y, sr = _load_mono_float32(input_wav, sr=ANALYSIS_SR)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    _save_float32_to_wav(y_shifted, sr, output_wav)
    return True


def _pitch_shift_hoat_ngon(input_wav: str, output_wav: str, n_steps: float) -> bool:
    """
    Pitch shift chuyên dụng cho giọng hoạt ngôn — tối ưu độ trong trẻo.

    Khác với _pitch_shift_clean:
    - pitchq=quality + transients=crisp + window=short: giữ rõ consonant,
      ít smearing hơn, sắc nét cho giọng nói nhanh
    - formant=shifted (mặc định, KHÔNG preserve) → formant tăng cùng pitch,
      tạo cảm giác giọng trẻ trung, tươi tắn tự nhiên (giống CapCut Hoạt Ngôn)
    - Thêm chuỗi EQ sau pitch shift để tăng độ trong trẻo:
        • highpass=80Hz  → loại bỏ tiếng ầm ì / rumble thấp
        • equalizer 300Hz -2dB (width_type=o, width=1.0) → giảm muddy
        • equalizer 3000Hz +3dB (width_type=o, width=1.5) → tăng presence/clarity
        • equalizer 8000Hz +2dB (width_type=o, width=2.0) → thêm air/sparkle

    Thứ tự ưu tiên:
    1. FFmpeg rubberband (quality + crisp transients + short window) + EQ
    2. Resampling + EQ (fallback)
    3. Librosa (fallback cuối)
    """
    if abs(n_steps) < 0.1:
        import shutil
        if input_wav != output_wav:
            shutil.copy2(input_wav, output_wav)
        return True

    pitch_ratio = 2.0 ** (n_steps / 12.0)

    # EQ chuỗi để tăng độ trong trẻo sau khi pitch shift
    # highpass cắt rumble, giảm muddy 300Hz, boost presence 3kHz, boost air 8kHz
    eq_chain = (
        "highpass=f=80,"
        "equalizer=f=300:width_type=o:width=1.0:g=-2,"
        "equalizer=f=3000:width_type=o:width=1.5:g=3,"
        "equalizer=f=8000:width_type=o:width=2.0:g=2"
    )

    # 1. Rubberband quality + EQ
    # - pitchq=quality: chất lượng cao nhất
    # - transients=crisp: giữ rõ consonant (s, t, ch...) — rất quan trọng cho giọng trong
    # - formant=shifted (mặc định): formant tăng theo pitch → giọng trẻ tự nhiên như CapCut
    # - window=short: cửa sổ ngắn → ít smearing hơn cho giọng nói nhanh
    af_rubberband_eq = f"rubberband=pitch={pitch_ratio:.6f}:pitchq=quality:transients=crisp:window=short,{eq_chain}"
    cmd_rubberband = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", input_wav,
        "-af", af_rubberband_eq,
        output_wav
    ]
    try:
        res = subprocess.run(cmd_rubberband, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 100:
            return True
    except Exception:
        pass

    # 2. Resampling + EQ fallback
    try:
        tempo_ratio = 1.0 / pitch_ratio
        sr = 24000
        try:
            with wave.open(input_wav, "rb") as wf:
                sr = wf.getframerate()
        except Exception:
            pass

        target_rate = int(sr * pitch_ratio)
        af_resample_eq = f"asetrate={target_rate},aresample={sr},atempo={tempo_ratio:.6f},{eq_chain}"
        cmd_resample = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", input_wav,
            "-af", af_resample_eq,
            output_wav
        ]
        res = subprocess.run(cmd_resample, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 100:
            return True
    except Exception:
        pass

    # 3. Fallback: Librosa (không EQ — vẫn tốt hơn không shift)
    librosa = _import_librosa()
    y, sr = _load_mono_float32(input_wav, sr=ANALYSIS_SR)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    _save_float32_to_wav(y_shifted, sr, output_wav)
    return True


def apply_pitch_shift(
    tts_wav_path: str,
    orig_wav_path: str,
    orig_start_sec: float,
    orig_end_sec: float,
    output_path: str,
    max_shift_semitones: float = 6.0,
) -> bool:
    """
    Phương án 1: Simple Pitch Shift.

    Phân tích pitch trung bình của giọng gốc trong [orig_start_sec, orig_end_sec],
    rồi shift pitch TTS output theo delta semitones tương ứng.

    Args:
        tts_wav_path: Đường dẫn file WAV TTS output.
        orig_wav_path: Đường dẫn file WAV âm thanh gốc.
        orig_start_sec: Thời điểm bắt đầu đoạn gốc (giây).
        orig_end_sec: Thời điểm kết thúc đoạn gốc (giây).
        output_path: Đường dẫn xuất kết quả.
        max_shift_semitones: Giới hạn tối đa số semitones shift (tránh biến dạng).

    Returns:
        True nếu thành công, False nếu thất bại (caller dùng TTS gốc).
    """
    try:
        # 1. Lấy pitch trung bình gốc
        orig_f0 = _extract_mean_pitch_librosa(orig_wav_path, orig_start_sec, orig_end_sec)

        # 2. Lấy pitch trung bình TTS
        tts_f0 = _extract_mean_pitch_librosa(tts_wav_path)

        if orig_f0 <= 0 or tts_f0 <= 0:
            # Không detect được pitch (đoạn silence hoặc noise) → bỏ qua
            if output_path != tts_wav_path:
                import shutil
                shutil.copy2(tts_wav_path, output_path)
            return True

        # 3. Tính delta semitones
        delta_semitones = _hz_to_semitones(orig_f0, tts_f0)

        # 4. Giới hạn shift để tránh biến dạng quá mức
        delta_semitones = np.clip(delta_semitones, -max_shift_semitones, max_shift_semitones)

        if abs(delta_semitones) < 0.5:
            # Quá nhỏ → không cần shift
            import shutil
            shutil.copy2(tts_wav_path, output_path)
            return True

        # 5. Shift bằng thuật toán Rubberband sạch tiếng (không vang)
        ok = _pitch_shift_clean(tts_wav_path, output_path, delta_semitones)

        if ok:
            print(f"           [Pitch-Shift] Orig={orig_f0:.1f}Hz TTS={tts_f0:.1f}Hz "
                  f"Δ={delta_semitones:+.1f} semitones → {os.path.basename(output_path)}")
        return ok

    except Exception as e:
        print(f"           [Pitch-Shift] ⚠ Lỗi: {e}. Dùng TTS gốc.")
        return False


# ─── Phương án 2: F0 Contour Cloning ─────────────────────────────────────────

def _extract_f0_pyworld(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dùng pyworld để phân tích WORLD vocoder parameters: F0, SP, AP.
    Trả về (f0, sp, ap) arrays.
    """
    pw = _import_pyworld()

    # pyworld cần float64 và sample rate hỗ trợ (thường 16000 hoặc 22050)
    y64 = y.astype(np.float64)

    # Normalize để tránh underflow
    max_amp = np.max(np.abs(y64))
    if max_amp > 0:
        y64 = y64 / max_amp * 0.9

    # DIO — fast F0 estimator
    _f0, t = pw.dio(y64, sr, f0_floor=65.0, f0_ceil=800.0,
                    channels_in_octave=2, frame_period=5.0)
    f0 = pw.stonemask(y64, _f0, t, sr)  # refine với StoneMask

    # Spectral Envelope và Aperiodicity
    sp = pw.cheaptrick(y64, f0, t, sr)
    ap = pw.d4c(y64, f0, t, sr)

    return f0, sp, ap


def _resample_contour(contour: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample một mảng F0 contour về độ dài target_len dùng interpolation tuyến tính.
    Giữ nguyên tỷ lệ voiced/unvoiced.
    """
    src_len = len(contour)
    if src_len == target_len:
        return contour.copy()

    # Tách voiced mask trước khi interpolate
    voiced_mask = contour > VOICED_THRESHOLD

    # Chỉ interpolate trên các điểm voiced, unvoiced giữ là 0
    x_old = np.linspace(0, 1, src_len)
    x_new = np.linspace(0, 1, target_len)

    # Interpolate toàn bộ (kể cả 0) rồi áp lại mask
    resampled = np.interp(x_new, x_old, contour)

    # Phục hồi unvoiced regions: dùng voiced_mask resample
    voiced_mask_resampled = np.interp(x_new, x_old, voiced_mask.astype(float)) > 0.5
    resampled[~voiced_mask_resampled] = 0.0

    return resampled


def _scale_f0_contour(orig_f0: np.ndarray, tts_f0: np.ndarray) -> np.ndarray:
    """
    Warp orig_f0 contour vào pitch space của TTS.
    Giữ nguyên hình dạng (lên/xuống) của orig, nhưng điều chỉnh
    về khoảng pitch của TTS để tránh chênh lệch quá lớn (giữ giọng natural).

    Công thức:
        new_f0[t] = tts_mean * (orig_f0[t] / orig_mean)  nếu voiced
                  = 0                                     nếu unvoiced
    """
    orig_mean = _median_voiced_f0(orig_f0)
    tts_mean  = _median_voiced_f0(tts_f0)

    if orig_mean <= 0 or tts_mean <= 0:
        return tts_f0.copy()

    new_f0 = np.zeros_like(tts_f0)
    voiced = tts_f0 > VOICED_THRESHOLD

    # Chuẩn hóa orig contour về relative (tỷ lệ so với mean)
    orig_relative = np.zeros_like(orig_f0)
    orig_voiced   = orig_f0 > VOICED_THRESHOLD
    orig_relative[orig_voiced] = orig_f0[orig_voiced] / orig_mean

    # Resample orig_relative về độ dài TTS
    orig_relative_resampled = _resample_contour(orig_relative, len(tts_f0))

    # Áp dụng: new_f0 = tts_mean * orig_relative (trong voiced regions của TTS)
    new_f0[voiced] = tts_mean * orig_relative_resampled[voiced]

    # Đảm bảo F0 trong khoảng hợp lệ
    new_f0[voiced] = np.clip(new_f0[voiced], 65.0, 800.0)

    return new_f0


def apply_f0_contour_clone(
    tts_wav_path: str,
    orig_wav_path: str,
    orig_start_sec: float,
    orig_end_sec: float,
    output_path: str,
    blend_ratio: float = 0.7,
) -> bool:
    """
    Phương án 2: F0 Contour Cloning dùng WORLD vocoder (pyworld).

    Extract đường cong F0 từ audio gốc, warp vào không gian pitch của TTS,
    rồi resynthesis audio với F0 mới. Giữ nguyên spectral envelope (timbre)
    của TTS (âm sắc giọng đọc), chỉ thay đổi intonation pattern.

    Args:
        tts_wav_path: Đường dẫn file WAV TTS output.
        orig_wav_path: Đường dẫn file WAV âm thanh gốc.
        orig_start_sec: Thời điểm bắt đầu đoạn gốc (giây).
        orig_end_sec: Thời điểm kết thúc đoạn gốc (giây).
        output_path: Đường dẫn xuất kết quả.
        blend_ratio: Tỷ lệ trộn F0 mới (0.0=giữ TTS, 1.0=dùng hoàn toàn orig contour).

    Returns:
        True nếu thành công, False nếu thất bại.
    """
    pw = _import_pyworld()

    try:
        # ── 1. Load audio ────────────────────────────────────────────────────
        # Dùng pyworld SR (16kHz) để phân tích F0 tốt hơn
        librosa = _import_librosa()

        y_orig_full, sr = librosa.load(orig_wav_path, sr=PYWORLD_SR, mono=True)
        y_tts, _        = librosa.load(tts_wav_path,  sr=PYWORLD_SR, mono=True)

        # Cắt đoạn gốc tương ứng với segment
        s_start = int(orig_start_sec * sr)
        s_end   = int(orig_end_sec   * sr)
        s_end   = min(s_end, len(y_orig_full))
        y_orig  = y_orig_full[s_start:s_end]

        if len(y_orig) < sr * 0.1:  # Quá ngắn
            import shutil
            shutil.copy2(tts_wav_path, output_path)
            return True

        # ── 2. WORLD Analysis ─────────────────────────────────────────────────
        f0_orig, _, _        = _extract_f0_pyworld(y_orig, sr)
        f0_tts, sp_tts, ap_tts = _extract_f0_pyworld(y_tts, sr)

        # ── 3. Warp F0 contour ───────────────────────────────────────────────
        f0_new = _scale_f0_contour(f0_orig, f0_tts)

        # Blend: trộn giữa F0 gốc của TTS và F0 clone mới
        voiced_mask = f0_tts > VOICED_THRESHOLD
        f0_blended  = f0_tts.copy()
        f0_blended[voiced_mask] = (
            (1.0 - blend_ratio) * f0_tts[voiced_mask] +
            blend_ratio * f0_new[voiced_mask]
        )
        # Clip range
        f0_blended[voiced_mask] = np.clip(f0_blended[voiced_mask], 65.0, 800.0)

        # ── 4. WORLD Synthesis ───────────────────────────────────────────────
        y_synth = pw.synthesize(
            f0_blended.astype(np.float64),
            sp_tts.astype(np.float64),
            ap_tts.astype(np.float64),
            sr,
            frame_period=5.0
        )

        # Normalize amplitude về mức của TTS gốc
        tts_rms  = np.sqrt(np.mean(y_tts.astype(np.float64) ** 2))
        synth_rms = np.sqrt(np.mean(y_synth ** 2))
        if synth_rms > 1e-6:
            y_synth = y_synth * (tts_rms / synth_rms)

        y_synth = y_synth.astype(np.float32)

        # ── 5. Lưu kết quả ───────────────────────────────────────────────────
        _save_float32_to_wav(y_synth, sr, output_path)

        orig_mean_hz = _median_voiced_f0(f0_orig)
        tts_mean_hz  = _median_voiced_f0(f0_tts)
        print(f"           [F0-Clone] Orig={orig_mean_hz:.1f}Hz TTS={tts_mean_hz:.1f}Hz "
              f"blend={blend_ratio:.0%} → {os.path.basename(output_path)}")
        return True

    except Exception as e:
        print(f"           [F0-Clone] ⚠ Lỗi: {e}. Dùng TTS gốc.")
        return False


def apply_hoat_ngon_pitch_shift(
    tts_wav_path: str,
    output_path: str,
    n_steps: float = 3.5,
) -> bool:
    """
    Phương án 3: Biến đổi giọng TTS theo phong cách 'Cô gái hoạt ngôn' (CapCut style).
    Đẩy cao độ (pitch) lên ~+3.5 semitones để tạo tông giọng tươi trẻ, nhanh nhẹn và hoạt ngôn.
    Không phụ thuộc vào file âm thanh gốc.

    Cải thiện v2 — Tăng độ trong trẻo:
    - Dùng rubberband pitchq=crisp (sắc nét, ít smearing hơn quality)
    - Formant tăng tự nhiên cùng pitch (không preserve) → giọng trẻ như CapCut
    - EQ sau pitch: loại rumble thấp, giảm muddy, boost presence 3kHz, boost air 8kHz
    """
    try:
        ok = _pitch_shift_hoat_ngon(tts_wav_path, output_path, n_steps=n_steps)
        if ok:
            print(f"           [Hoạt-Ngôn] Shift +{n_steps:.1f} semitones + EQ trong trẻo → {os.path.basename(output_path)}")
        return ok
    except Exception as e:
        print(f"           [Hoạt-Ngôn] ⚠ Lỗi: {e}. Dùng TTS gốc.")
        import shutil
        if tts_wav_path != output_path:
            shutil.copy2(tts_wav_path, output_path)
        return False


# ─── Entry Point Chính ────────────────────────────────────────────────────────

def apply_pitch_to_segment(
    tts_wav_path: str,
    orig_wav_path: str,
    orig_start_sec: float,
    orig_end_sec: float,
    output_path: str,
    method: str = "shift",         # "shift" | "clone" | "hoat_ngon" | "none"
    max_shift_semitones: float = 6.0,
    f0_blend_ratio: float = 0.7,
) -> bool:
    """
    Áp dụng xử lý cao độ lên TTS audio dựa theo âm thanh gốc hoặc preset.

    Args:
        tts_wav_path: WAV file từ TTS (input).
        orig_wav_path: WAV file âm thanh gốc của video.
        orig_start_sec: Thời điểm bắt đầu segment trong âm thanh gốc (giây).
        orig_end_sec: Thời điểm kết thúc segment trong âm thanh gốc (giây).
        output_path: Đường dẫn lưu WAV đã xử lý pitch (output).
        method: Phương pháp:
            "shift"     → Phương án 1: Simple Pitch Shift (match pitch gốc)
            "clone"     → Phương án 2: F0 Contour Cloning (nhại ngữ điệu)
            "hoat_ngon" → Phương án 3: Giọng Cô gái hoạt ngôn CapCut (+3.5 semitones)
            "none"      → Bỏ qua, copy thẳng
        max_shift_semitones: Giới hạn shift cho method="shift".
        f0_blend_ratio: Tỷ lệ blend F0 cho method="clone".

    Returns:
        True nếu thành công, False nếu thất bại.
    """
    if method == "none":
        import shutil
        shutil.copy2(tts_wav_path, output_path)
        return True

    if method in ["hoat_ngon", "hoat-ngon"]:
        return apply_hoat_ngon_pitch_shift(tts_wav_path, output_path, n_steps=3.5)

    if not os.path.exists(orig_wav_path):
        print(f"           [Pitch] ⚠ Không tìm thấy orig audio: {orig_wav_path}")
        import shutil
        shutil.copy2(tts_wav_path, output_path)
        return False

    if method == "shift":
        return apply_pitch_shift(
            tts_wav_path, orig_wav_path,
            orig_start_sec, orig_end_sec,
            output_path,
            max_shift_semitones=max_shift_semitones,
        )
    elif method == "clone":
        success = apply_f0_contour_clone(
            tts_wav_path, orig_wav_path,
            orig_start_sec, orig_end_sec,
            output_path,
            blend_ratio=f0_blend_ratio,
        )
        if not success:
            # Fallback về method 1 nếu clone thất bại
            print(f"           [F0-Clone] Fallback → Pitch Shift")
            return apply_pitch_shift(
                tts_wav_path, orig_wav_path,
                orig_start_sec, orig_end_sec,
                output_path,
                max_shift_semitones=max_shift_semitones,
            )
        return success
    else:
        print(f"           [Pitch] ⚠ Method không hợp lệ: '{method}'. Dùng 'shift' | 'clone' | 'hoat_ngon' | 'none'.")
        import shutil
        shutil.copy2(tts_wav_path, output_path)
        return False


def apply_pitch_to_all_segments(
    segments: list,
    orig_audio_path: str,
    method: str = "shift",
    max_shift_semitones: float = 6.0,
    f0_blend_ratio: float = 0.7,
) -> list:
    """
    Áp dụng pitch processing cho tất cả segments sau khi TTS đã generate xong.
    Cập nhật segment['tts_audio_path'] trỏ sang file đã xử lý pitch.

    Args:
        segments: List segment dicts (đã có 'tts_audio_path', 'start', 'end').
        orig_audio_path: Đường dẫn WAV âm thanh gốc của video.
        method: "shift" | "clone" | "none"
        max_shift_semitones: Giới hạn shift (cho method="shift").
        f0_blend_ratio: Blend ratio (cho method="clone").

    Returns:
        segments đã được cập nhật.
    """
    if method == "none":
        return segments

    import time
    print(f"\n[Module 4.5] Pitch processing ({method}) cho {len(segments)} segments...")
    t0 = time.time()

    success_count = 0
    skip_count    = 0
    fail_count    = 0

    for segment in segments:
        tts_path = segment.get("tts_audio_path")

        if not tts_path or not os.path.exists(tts_path):
            skip_count += 1
            continue

        orig_start = segment.get("start", 0.0)
        orig_end   = segment.get("end",   orig_start + 0.5)

        # Output: cùng thư mục, thêm hậu tố _pitched
        pitched_path = tts_path.replace(".wav", f"_p{method[:1]}.wav")

        ok = apply_pitch_to_segment(
            tts_wav_path=tts_path,
            orig_wav_path=orig_audio_path,
            orig_start_sec=float(orig_start),
            orig_end_sec=float(orig_end),
            output_path=pitched_path,
            method=method,
            max_shift_semitones=max_shift_semitones,
            f0_blend_ratio=f0_blend_ratio,
        )

        if ok and os.path.exists(pitched_path) and os.path.getsize(pitched_path) > 100:
            from backend.app.pipeline.audio_utils import get_wav_duration_fast
            segment["tts_audio_path"] = pitched_path
            new_dur = get_wav_duration_fast(pitched_path)
            segment["tts_duration"] = new_dur
            segment["dur_ms"] = int(new_dur * 1000)
            success_count += 1
        else:
            # Giữ nguyên TTS gốc nếu thất bại
            fail_count += 1

    elapsed = time.time() - t0
    print(f"[Module 4.5] Xong: {success_count} thành công, {skip_count} bỏ qua, "
          f"{fail_count} thất bại | {elapsed:.1f}s")
    return segments


# ─── CLI Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test thủ công:
        python -m backend.app.pipeline.pitch \
            --tts output/test_tts/tts_1.wav \
            --orig output/original_audio.wav \
            --start 1.2 --end 3.5 \
            --method shift \
            --out output/test_pitched.wav
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test pitch processing module")
    parser.add_argument("--tts",    required=True, help="TTS WAV file")
    parser.add_argument("--orig",   required=True, help="Original audio WAV file")
    parser.add_argument("--start",  type=float, default=0.0,  help="Segment start (sec)")
    parser.add_argument("--end",    type=float, default=None,  help="Segment end (sec)")
    parser.add_argument("--method", default="shift", choices=["shift", "clone", "none"])
    parser.add_argument("--out",    required=True, help="Output WAV path")
    parser.add_argument("--max-shift", type=float, default=6.0)
    parser.add_argument("--blend",     type=float, default=0.7)
    args = parser.parse_args()

    end_sec = args.end
    if end_sec is None:
        # Lấy duration của orig audio
        try:
            with wave.open(args.orig, "rb") as wf:
                end_sec = wf.getnframes() / wf.getframerate()
        except Exception:
            end_sec = args.start + 5.0

    print(f"[Test] method={args.method} | start={args.start}s end={end_sec}s")
    ok = apply_pitch_to_segment(
        tts_wav_path=args.tts,
        orig_wav_path=args.orig,
        orig_start_sec=args.start,
        orig_end_sec=end_sec,
        output_path=args.out,
        method=args.method,
        max_shift_semitones=args.max_shift,
        f0_blend_ratio=args.blend,
    )
    print(f"[Test] {'✓ Thành công' if ok else '✗ Thất bại'} → {args.out}")
