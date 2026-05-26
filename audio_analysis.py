"""
audio_analysis.py
-----------------
Ses sinyalinden özellik çıkarma – Final Proje (Faz 3)
BIL216 – Grup 05 | 2025-2026 Bahar Dönemi

Özellik seti (~80 özellik):
  Zaman    : mean_f0, std_f0, mean_zcr, mean_energy, voiced_ratio, rms_mean, rms_std, tempo  (8)
  MFCC     : 13 mean + 13 std                                                                 (26)
  ΔMFCC    : 13 mean + 13 std                                                                 (26)
  Spektral : centroid, bw, rolloff, flatness (mean+std)                                        (8)
  Chroma   : 12 mean                                                                           (12)
  ──────────────────────────────────────────────────────────────────────────────────────────  80

Faz 3 Yenilikleri:
  - Ses düzeyinde veri artırma: pitch_shift, time_stretch, add_noise
    (classifier.py içinde eğitim öncesi uygulanır)
"""

import numpy as np
import librosa

N_MFCC = 13


def load_audio(filepath, sr=22050):
    audio, sr = librosa.load(filepath, sr=sr, mono=True)
    return audio, sr


def get_frames(audio, sr, frame_ms=25, hop_ms=10):
    frame_length = int(sr * frame_ms / 1000)
    hop_length   = int(sr * hop_ms  / 1000)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    return frames, frame_length, hop_length


def compute_energy(frames):
    return np.sum(frames ** 2, axis=0)


def compute_zcr(frames):
    signs     = np.sign(frames)
    sign_diff = np.diff(signs, axis=0)
    return np.sum(np.abs(sign_diff), axis=0) / (2 * frames.shape[0])


def detect_voiced_frames(energy, zcr,
                          energy_threshold_ratio=0.05,
                          zcr_threshold=0.15):
    thr = energy_threshold_ratio * np.max(energy)
    return (energy > thr) & (zcr < zcr_threshold)


def autocorrelation_f0(frame, sr, f0_min=50, f0_max=500):
    n       = len(frame)
    lag_min = int(sr / f0_max)
    lag_max = min(int(sr / f0_min), n - 1)
    if lag_min >= lag_max:
        return 0.0
    autocorr = np.correlate(frame, frame, mode='full')[n - 1:]
    segment  = autocorr[lag_min:lag_max]
    if len(segment) == 0 or np.max(segment) == 0:
        return 0.0
    peak_lag = np.argmax(segment) + lag_min
    return sr / peak_lag if peak_lag > 0 else 0.0


def get_autocorr_array(frame):
    n        = len(frame)
    autocorr = np.correlate(frame, frame, mode='full')[n - 1:]
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]
    return autocorr


def compute_fft(frame, sr):
    N        = len(frame)
    windowed = frame * np.hanning(N)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs    = np.fft.rfftfreq(N, d=1.0 / sr)
    return freqs, spectrum


# ──────────────────────────────────────────────
# MFCC + DELTA MFCC
# ──────────────────────────────────────────────

def compute_mfcc(audio, sr, n_mfcc=N_MFCC):
    mfcc       = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    return (np.mean(mfcc,       axis=1), np.std(mfcc,       axis=1),
            np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1))


def mfcc_column_names(n_mfcc=N_MFCC):
    return ([f"mfcc_mean_{i}"  for i in range(n_mfcc)] +
            [f"mfcc_std_{i}"   for i in range(n_mfcc)])


def delta_mfcc_column_names(n_mfcc=N_MFCC):
    return ([f"delta_mean_{i}" for i in range(n_mfcc)] +
            [f"delta_std_{i}"  for i in range(n_mfcc)])


# ──────────────────────────────────────────────
# SPEKTRAL + CHROMA + RMS + TEMPO
# ──────────────────────────────────────────────

def compute_spectral_features(audio, sr):
    centroid  = librosa.feature.spectral_centroid(y=audio,  sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=audio,   sr=sr)[0]
    flatness  = librosa.feature.spectral_flatness(y=audio)[0]
    return {
        "spec_centroid_mean"  : float(np.mean(centroid)),
        "spec_centroid_std"   : float(np.std(centroid)),
        "spec_bandwidth_mean" : float(np.mean(bandwidth)),
        "spec_bandwidth_std"  : float(np.std(bandwidth)),
        "spec_rolloff_mean"   : float(np.mean(rolloff)),
        "spec_rolloff_std"    : float(np.std(rolloff)),
        "spec_flatness_mean"  : float(np.mean(flatness)),
        "spec_flatness_std"   : float(np.std(flatness)),
    }


SPECTRAL_COLS = [
    "spec_centroid_mean","spec_centroid_std",
    "spec_bandwidth_mean","spec_bandwidth_std",
    "spec_rolloff_mean","spec_rolloff_std",
    "spec_flatness_mean","spec_flatness_std",
]


def compute_chroma(audio, sr):
    return np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)


def compute_rms(audio, sr):
    rms = librosa.feature.rms(y=audio)[0]
    return float(np.mean(rms)), float(np.std(rms))


def compute_tempo(audio, sr):
    try:
        tempo = librosa.beat.beat_track(y=audio, sr=sr)[0]
        return float(np.atleast_1d(tempo)[0])
    except Exception:
        return 0.0


def chroma_column_names():
    return [f"chroma_{i}" for i in range(12)]


# ──────────────────────────────────────────────
# SES DÜZEYİNDE VERİ ARTIRMA
# ──────────────────────────────────────────────

def augment_audio(audio: np.ndarray, sr: int, mode: str) -> np.ndarray:
    """
    Ham ses sinyaline artırma uygular.

    mode:
      'noise'   : hafif beyaz gürültü ekle
      'pitch+'  : yarım ton yukarı kaydır
      'pitch-'  : yarım ton aşağı kaydır
      'stretch' : %10 yavaşlat (zaman gerdirme)
    """
    try:
        if mode == "noise":
            noise = np.random.randn(len(audio)) * 0.003 * np.std(audio)
            return (audio + noise).astype(np.float32)
        elif mode == "pitch+":
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
        elif mode == "pitch-":
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=-0.5)
        elif mode == "stretch":
            stretched = librosa.effects.time_stretch(audio, rate=0.9)
            # Orijinal uzunluğa geri döndür
            if len(stretched) > len(audio):
                return stretched[:len(audio)]
            else:
                return np.pad(stretched, (0, len(audio) - len(stretched)))
    except Exception:
        pass
    return audio


# ──────────────────────────────────────────────
# ÖZELLİK ÇIKARIMI (TEK SES)
# ──────────────────────────────────────────────

def _extract_from_signal(audio: np.ndarray, sr: int) -> dict:
    """Ham ses sinyalinden özellik çıkarır (dosya yolu olmadan)."""
    frames, _, hop_length = get_frames(audio, sr)
    energy      = compute_energy(frames)
    zcr         = compute_zcr(frames)
    voiced_mask = detect_voiced_frames(energy, zcr)

    f0_values = []
    for i, is_voiced in enumerate(voiced_mask):
        if is_voiced:
            f0 = autocorrelation_f0(frames[:, i], sr)
            if f0 > 0:
                f0_values.append(f0)

    f0_values    = np.array(f0_values)
    mean_f0      = float(np.mean(f0_values))  if len(f0_values) > 0 else 0.0
    std_f0       = float(np.std(f0_values))   if len(f0_values) > 0 else 0.0
    mean_zcr     = float(np.mean(zcr))
    mean_energy  = float(np.mean(energy))
    voiced_ratio = float(np.sum(voiced_mask) / len(voiced_mask))

    mfcc_mean, mfcc_std, delta_mean, delta_std = compute_mfcc(audio, sr)
    spectral    = compute_spectral_features(audio, sr)
    chroma_mean = compute_chroma(audio, sr)
    rms_mean, rms_std = compute_rms(audio, sr)
    tempo       = compute_tempo(audio, sr)

    return {
        "mean_f0"     : mean_f0,
        "std_f0"      : std_f0,
        "mean_zcr"    : mean_zcr,
        "mean_energy" : mean_energy,
        "voiced_ratio": voiced_ratio,
        "rms_mean"    : rms_mean,
        "rms_std"     : rms_std,
        "tempo"       : tempo,
        "mfcc_mean"   : mfcc_mean,
        "mfcc_std"    : mfcc_std,
        "delta_mean"  : delta_mean,
        "delta_std"   : delta_std,
        **spectral,
        "chroma_mean" : chroma_mean,
        "hop_length"  : hop_length,
        "energy_arr"  : energy,
        "zcr_arr"     : zcr,
        "voiced_mask" : voiced_mask,
        "sample_frame": frames[:, frames.shape[1] // 2],
        "f0_values"   : f0_values,
        "audio"       : audio,
        "sr"          : sr,
    }


def extract_features(filepath: str, sr: int = 22050) -> dict:
    """Dosyadan özellik çıkarır."""
    audio, sr = load_audio(filepath, sr=sr)
    return _extract_from_signal(audio, sr)


def extract_features_augmented(filepath: str,
                                sr: int = 22050,
                                modes: list = None) -> list:
    """
    Orijinal + artırılmış ses versiyonlarından özellik çıkarır.
    Her versiyon için ayrı bir dict döndürür.

    modes: ['noise','pitch+','pitch-','stretch'] (None → hepsi)
    """
    if modes is None:
        modes = ["noise", "pitch+", "pitch-", "stretch"]

    audio, sr = load_audio(filepath, sr=sr)
    results   = [_extract_from_signal(audio, sr)]  # orijinal

    for mode in modes:
        aug_audio = augment_audio(audio, sr, mode)
        try:
            results.append(_extract_from_signal(aug_audio, sr))
        except Exception:
            pass

    return results
