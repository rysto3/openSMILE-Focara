from __future__ import annotations
from typing import Dict, List, Iterable, Any
import argparse
import json
import sys
import time
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import librosa
import opensmile

"""
Batch activity classifier for music tracks with CSV + JSON outputs.

Fixes in v2:
- Robust conversion of NumPy arrays/scalars to JSON-safe Python types (no ndarray errors)
- Safer _to_float that accepts 0-d arrays and numpy scalars (removes deprecation warning)
- Ensures features.csv is truly wide (MFCCs expanded; no list/array cells)

Outputs (created/updated in the working directory):
  - scores.csv   : rows -> [track, Focus, Calming, Sleep, Other]
  - scores.json  : JSON array of the same rows
  - features.csv : wide table -> [track, <feature1>, <feature2>, ...]
  - features.json: JSON array of the same rows

Usage examples:
  python activity_classifier_batch_v2.py                     # processes ./smile.wav if present
  python activity_classifier_batch_v2.py path/to/song.wav
  python activity_classifier_batch_v2.py path/to/folder      # processes all *.wav recursively
  python activity_classifier_batch_v2.py path --no-recursive # folder, but non-recursive
"""

# -------------------------------
# JSON / scalar coercion helpers
# -------------------------------

def to_python_scalar(x: Any) -> Any:
    """Convert NumPy/pandas objects to plain Python types for CSV/JSON.
    - 0-d arrays -> float/int
    - NumPy scalars -> .item()
    - NaN/NaT -> None
    - 1+-d arrays/lists -> list of python scalars
    """
    # pandas NA
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # numpy scalar
    if isinstance(x, np.generic):
        return x.item()

    # numpy array
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.reshape(()).item()
        return [to_python_scalar(v) for v in x.tolist()]

    # sequences
    if isinstance(x, (list, tuple)):
        return [to_python_scalar(v) for v in x]

    return x


def df_json_safe(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(to_python_scalar)


# -------------------------------
# Feature extraction
# -------------------------------

def extract_features(
    wav_path: str,
    smile: opensmile.Smile,
    lock: threading.Lock | None = None,
) -> pd.Series:
    # --- 1) Tempo & rhythm (librosa) ---
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(tempo)
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
    if lock is None:
        f = smile.process_signal(y, sr)  # 1 row
    else:
        with lock:
            f = smile.process_signal(y, sr)

    def pick(regex: str) -> float:
        cols = f.filter(regex=regex).columns
        return float(f.iloc[0][cols[0]]) if len(cols) else np.nan

    voicing_mean = pick(r'^VoicedSegmentsPerSec$') / 10.0  # scale roughly to 0-1
    voicing_p95 = pick(r'^MeanVoicedSegmentLengthSec$')
    f0_std = pick(r'^F0semitoneFrom27\.5Hz_.*_stddevNorm$')
    flux_median = pick(r'^spectralFlux_sma3_amean$') / 100.0  # scale to ~0-0.05

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
    # Ensure plain Python scalars in the series
    features = {k: to_python_scalar(v) for k, v in features.items()}
    return pd.Series(features)


# -------------------------------
# Interpretable classifier
# -------------------------------

def _to_float(x, default=np.nan):
    # Accept plain numbers, numpy scalars, and 0-d arrays without warnings
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            try:
                return float(x.reshape(()).item())
            except Exception:
                return default
        # Non-scalar arrays cannot be a single float
        return default
    if isinstance(x, np.generic):
        try:
            return float(x.item())
        except Exception:
            return default
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


# -------------------------------
# I/O helpers
# -------------------------------

def _load_existing_table(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return pd.DataFrame()


def _write_csv_json(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    df = df_json_safe(df)
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _gather_wavs(path: Path, recursive: bool = True) -> List[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted([p for p in path.glob(pattern) if p.is_file()])


# -------------------------------
# Main
# -------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch music activity classifier (CSV + JSON outputs)")
    parser.add_argument("path", nargs="?", default="smile.wav", help="Path to a WAV file or a folder")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not recurse into subfolders")
    args = parser.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        print(f"Error: not found -> {target}")
        return 2

    wavs = _gather_wavs(target, recursive=args.recursive)
    if not wavs:
        print("No WAV files found to process.")
        return 0

    t0 = time.perf_counter()

    # Load existing outputs (to merge/overwrite rows by `track`)
    scores_csv = Path("scores.csv")
    features_csv = Path("features.csv")
    scores_df = _load_existing_table(scores_csv)
    features_df = _load_existing_table(features_csv)

    new_score_rows: List[dict] = []
    new_feature_rows: List[dict] = []

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    smile_lock = threading.Lock()

    for i, wav in enumerate(wavs, 1):
        try:
            feat = extract_features(str(wav), smile, smile_lock)

            # Feature row (wide), first column is track
            feature_row = {"track": wav.name}
            for k, v in feat.items():
                feature_row[k] = to_python_scalar(v)
            new_feature_rows.append(feature_row)

            # Score row
            scores = classify_activity(feat)
            score_row = {
                "track": wav.name,
                "Focus": float(scores["Focus"]),
                "Calming": float(scores["Calming"]),
                "Sleep": float(scores["Sleep"]),
                "Other": float(scores["Other"]),
            }
            new_score_rows.append(score_row)

            print(f"[{i}/{len(wavs)}] {wav.name} -> Focus {scores['Focus']:.2f}, Calming {scores['Calming']:.2f}, Sleep {scores['Sleep']:.2f}")
        except Exception as e:
            print(f"[{i}/{len(wavs)}] {wav.name} -> ERROR: {e}")

    # Build DataFrames
    new_scores_df = pd.DataFrame(new_score_rows)
    new_features_df = pd.DataFrame(new_feature_rows)

    # Merge with existing (outer on 'track', prefer new rows for processed tracks)
    if not scores_df.empty:
        scores_df = scores_df[~scores_df['track'].isin(new_scores_df['track'])]
    combined_scores = pd.concat([scores_df, new_scores_df], ignore_index=True)
    # Fixed column order
    combined_scores = combined_scores[["track", "Focus", "Calming", "Sleep", "Other"]]

    # Features: unify columns, prefer new rows on conflict
    if not features_df.empty:
        features_df = features_df[~features_df['track'].isin(new_features_df['track'])]
        combined_features = pd.concat([features_df, new_features_df], ignore_index=True)
    else:
        combined_features = new_features_df

    # Column order: track first, then sorted others for stability
    feature_cols = ["track"] + sorted([c for c in combined_features.columns if c != "track"])
    combined_features = combined_features[feature_cols]

    # Write CSV + JSON (arrays)
    _write_csv_json(combined_scores, scores_csv, Path("scores.json"))
    _write_csv_json(combined_features, features_csv, Path("features.json"))

    elapsed = time.perf_counter() - t0
    print(f"\nProcessed {len(new_scores_df)} file(s) in {elapsed:.2f}s")
    print("Wrote: scores.csv, scores.json, features.csv, features.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
