from __future__ import annotations
from typing import Dict, List
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import opensmile

"""
End-to-end classifier with two CSV outputs:

1) scores.csv   -> one row per track: [track, Focus, Calming, Sleep, Other]
2) features.csv -> wide table:       [track, <feature columns...>]

Usage:
  python activity_classifier_full_csv.py                # uses ./smile.wav
  python activity_classifier_full_csv.py path/to/file.wav
"""

# -------------------------------
# Feature extraction
# -------------------------------

def extract_features(wav_path: str) -> pd.Series:
    # --- 1) Tempo & rhythm (librosa) ---
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    ibi = np.diff(beat_times)  # inter-beat intervals (s)
    beat_irregularity = float(np.std(ibi)) if len(ibi) > 2 else np.nan

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_rate = float(len(onset_frames) / (len(y) / sr))

    # Energy & dynamics (librosa)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    dyn_range = float(np.percentile(rms, 95) - np.percentile(rms, 5))

    # Timbre/brightness (librosa)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_mean = float(np.mean(cent))
    cent_std = float(np.std(cent))

    roll85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    roll85_mean = float(np.mean(roll85))

    flat = librosa.feature.spectral_flatness(y=y)[0]
    flat_median = float(np.median(flat))

    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    high = S[freqs >= 4000].sum()
    total = S.sum() + 1e-9
    high_ratio = float(high / total)

    # MFCCs (librosa; compact summary)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = [float(np.mean(mfcc[i])) for i in range(1, 6)]  # MFCC 1..5
    mfcc_stds = [float(np.std(mfcc[i])) for i in range(1, 6)]

    # --- 2) Voicing & pitch steadiness (openSMILE functionals) ---
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,  # or eGeMAPSv02 for a small set
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    f = smile.process_file(wav_path)  # 1 row

    def pick(regex: str) -> float:
        cols = f.filter(regex=regex).columns
        return float(f.iloc[0][cols[0]]) if len(cols) else np.nan

    voicing_mean = pick(r'^voicingProbability_.*(_amean|_mean)$')
    voicing_p95 = pick(r'^voicingProbability_.*percentile95\.0$')
    f0_std = pick(r'^F0final_.*_stddev$')
    flux_median = pick(r'^spectralFlux_.*_(median|pctlrange25-75|quartile2)$')  # robust-ish fallback

    features = {
        "tempo_bpm": tempo,
        "beat_irregularity": beat_irregularity,
        "onset_rate_per_s": onset_rate,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "dynamic_range": dyn_range,
        "centroid_mean_hz": cent_mean,
        "centroid_std_hz": cent_std,
        "rolloff85_mean_hz": roll85_mean,
        "spectral_flatness_median": flat_median,
        "high_freq_ratio_>4k": high_ratio,
        # expand MFCCs into scalar columns so the table is wide & clean
        **{f"mfcc{i}_mean": mfcc_means[i-1] for i in range(1, 6)},
        **{f"mfcc{i}_std": mfcc_stds[i-1] for i in range(1, 6)},
        "voicing_mean": voicing_mean,
        "voicing_p95": voicing_p95,
        "f0_std": f0_std,
        "spectral_flux_median": flux_median,
        "sr": sr,
    }
    return pd.Series(features)


# -------------------------------
# Interpretable classifier
# -------------------------------

def _to_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def score_low(val: float, best_max: float, worst_max: float) -> float:
    v = _to_float(val, np.nan)
    if np.isnan(v):
        return 0.5
    if v <= best_max:
        return 1.0
    if v >= worst_max:
        return 0.0
    return 1.0 - (v - best_max) / (worst_max - best_max)


def score_band(val: float, best_lo: float, best_hi: float, worst_lo: float, worst_hi: float) -> float:
    v = _to_float(val, np.nan)
    if np.isnan(v):
        return 0.5
    if v < worst_lo or v > worst_hi:
        return 0.0
    if best_lo <= v <= best_hi:
        return 1.0
    if v < best_lo:
        return (v - worst_lo) / (best_lo - worst_lo)
    return (worst_hi - v) / (worst_hi - best_hi)


def _nz(series: pd.Series, key: str, default=np.nan) -> float:
    return _to_float(series.get(key, default), default)


def classify_activity(feat: pd.Series) -> Dict[str, float]:
    tempo = _nz(feat, "tempo_bpm")
    beat_irreg = _nz(feat, "beat_irregularity")
    onset = _nz(feat, "onset_rate_per_s")
    rms_std = _nz(feat, "rms_std")
    dyn = _nz(feat, "dynamic_range")
    cent = _nz(feat, "centroid_mean_hz")
    hf = _nz(feat, "high_freq_ratio_>4k")
    voice = _nz(feat, "voicing_mean")
    flux = _nz(feat, "spectral_flux_median")

    # Sleep
    sleep = 0.0
    sleep += score_low(tempo, best_max=70, worst_max=95) * 2.0
    sleep += score_low(onset, best_max=1.5, worst_max=3.0) * 2.0
    sleep += score_low(rms_std, best_max=0.05, worst_max=0.15) * 1.5
    sleep += score_low(dyn, best_max=0.20, worst_max=0.60) * 1.25
    sleep += score_low(cent, best_max=1200, worst_max=2500) * 1.25
    sleep += score_low(hf, best_max=0.10, worst_max=0.40) * 1.0
    sleep += score_low(voice, best_max=0.20, worst_max=0.60) * 1.25
    if not np.isnan(beat_irreg):
        sleep += score_low(beat_irreg, best_max=0.04, worst_max=0.12) * 0.5
    if not np.isnan(flux):
        sleep += score_low(flux, best_max=0.01, worst_max=0.05) * 0.5

    # Focus
    focus = 0.0
    focus += score_band(tempo, best_lo=70, best_hi=110, worst_lo=50, worst_hi=130) * 2.0
    if not np.isnan(beat_irreg):
        focus += score_low(beat_irreg, best_max=0.03, worst_max=0.12) * 1.5
    focus += score_band(onset, best_lo=2.0, best_hi=6.0, worst_lo=0.5, worst_hi=10.0) * 1.75
    focus += score_band(rms_std, best_lo=0.05, best_hi=0.12, worst_lo=0.01, worst_hi=0.25) * 1.25
    focus += score_band(cent, best_lo=1200, best_hi=2600, worst_lo=600, worst_hi=4200) * 1.25
    focus += score_low(voice, best_max=0.25, worst_max=0.70) * 1.5

    # Calming
    calming = 0.0
    calming += score_band(tempo, best_lo=55, best_hi=90, worst_lo=40, worst_hi=115) * 1.75
    calming += score_band(onset, best_lo=1.0, best_hi=3.0, worst_lo=0.2, worst_hi=6.0) * 1.75
    calming += score_low(rms_std, best_max=0.08, worst_max=0.20) * 1.25
    calming += score_low(dyn, best_max=0.30, worst_max=0.70) * 1.0
    calming += score_low(cent, best_max=1600, worst_max=3000) * 1.0
    calming += score_low(hf, best_max=0.18, worst_max=0.45) * 0.75
    calming += score_low(voice, best_max=0.30, worst_max=0.75) * 1.25
    if not np.isnan(flux):
        calming += score_low(flux, best_max=0.02, worst_max=0.06) * 0.5

    raw = np.array([sleep, focus, calming], dtype=float)
    raw = np.maximum(raw, 0.0)
    probs = raw / raw.sum() if raw.sum() > 0 else np.array([1/3, 1/3, 1/3])

    out = {"Sleep": float(probs[0]), "Focus": float(probs[1]), "Calming": float(probs[2])}
    out["Other"] = float(max(0.0, 1.0 - sum(out.values())))
    return out


def pretty_scores(scores: Dict[str, float]) -> str:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return "\n".join(f"{k:8s} : {v*100:5.1f}%" for k, v in items)


# -------------------------------
# CSV helpers
# -------------------------------

def append_row_csv(path: Path, row: dict, field_order: List[str] | None = None) -> None:
    df = pd.DataFrame([row])
    if field_order:
        # ensure any missing columns are present
        for col in field_order:
            if col not in df.columns:
                df[col] = np.nan
        df = df[field_order]
    header = not path.exists()
    df.to_csv(path, mode='a', header=header, index=False)


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "smile.wav"
    wav = Path(wav_path)
    if not wav.exists():
        print(f"Error: file not found -> {wav}")
        sys.exit(1)

    t0 = time.perf_counter()
    feat = extract_features(str(wav))

    # ---- build wide feature row (first column is track filename)
    feature_row = {"track": wav.name}
    for k, v in feat.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            # we already expanded MFCCs; this is a safeguard
            for i, x in enumerate(v):
                feature_row[f"{k}_{i}"] = float(x)
        else:
            feature_row[k] = float(v) if isinstance(v, (int, float, np.floating)) else v

    # Determine a stable feature column order (persist across runs)
    feature_cols = ["track"] + sorted([c for c in feature_row.keys() if c != "track"])

    # ---- classify & make scores row in order: Focus, Calming, Sleep, Other
    scores = classify_activity(feat)
    score_cols = ["track", "Focus", "Calming", "Sleep", "Other"]
    score_row = {"track": wav.name, **{k: float(scores.get(k, 0.0)) for k in score_cols if k != "track"}}

    # ---- write CSVs (append or create with header)
    append_row_csv(Path("features.csv"), feature_row, field_order=feature_cols)
    append_row_csv(Path("scores.csv"), score_row, field_order=score_cols)

    elapsed = time.perf_counter() - t0
    print(f"Processed {wav.name} in {elapsed:.3f}s\n")
    print(pretty_scores(scores))
    print("\nWrote: features.csv (wide), scores.csv (track + scores)")
