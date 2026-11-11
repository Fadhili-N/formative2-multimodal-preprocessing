"""Step 4: clean -> augment -> extract features -> audio_features.csv

Run:  python audio_pipeline.py
Every original clip becomes 1 original row + 6 augmented rows.
`clip_id` ties augmented rows back to the original recording so the
train/test split in voice_model.py never puts copies of one recording on both sides.
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).parent
WAV = ROOT / "converted"
AUG_DIR = ROOT / "augmented"
SR = 16000
N_MFCC = 13
RNG = np.random.default_rng(42)


# ---------- cleaning ----------
def clean(y: np.ndarray) -> np.ndarray:
    """Cut a clipped 'button tap' click at the end, trim silence, normalise volume."""
    clipped = np.where(np.abs(y) >= 0.99)[0]
    if len(clipped) and clipped[0] > 0.5 * len(y):      # click in the 2nd half = stop-button tap
        y = y[: max(clipped[0] - int(0.05 * SR), 1)]
    y, _ = librosa.effects.trim(y, top_db=30)             # drop leading/trailing silence
    return 0.9 * y / (np.abs(y).max() + 1e-9)             # peak-normalise


# ---------- augmentations ----------
def add_noise(y, snr_db):
    noise = RNG.standard_normal(len(y))
    scale = np.sqrt((y ** 2).mean() / (10 ** (snr_db / 10)) / (noise ** 2).mean())
    return y + scale * noise


AUGMENTATIONS = {
    "pitch_up":    lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=1.5),
    "pitch_down":  lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-1.5),
    "stretch_fast": lambda y: librosa.effects.time_stretch(y, rate=1.15),
    "stretch_slow": lambda y: librosa.effects.time_stretch(y, rate=0.85),
    "noise_20db":  lambda y: add_noise(y, 20),
    "noise_10db":  lambda y: add_noise(y, 10),
}


# ---------- features ----------
def extract(y: np.ndarray) -> dict:
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85)[0]
    rms = librosa.feature.rms(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    f = {}
    for i in range(N_MFCC):
        f[f"mfcc{i+1}_mean"] = mfcc[i].mean()
        f[f"mfcc{i+1}_std"] = mfcc[i].std()
    f.update(rolloff_mean=rolloff.mean(), rolloff_std=rolloff.std(),
             energy_mean=rms.mean(), energy_std=rms.std(),
             energy_total=float((y ** 2).sum()),
             centroid_mean=cent.mean(), bandwidth_mean=bw.mean(),
             zcr_mean=zcr.mean(), duration_s=len(y) / SR)
    return f


def main():
    AUG_DIR.mkdir(exist_ok=True)
    rows = []
    for path in sorted(WAV.glob("*.wav")):
        speaker, phrase = path.stem.split("_", 1)
        y, _ = librosa.load(path, sr=SR)
        y = clean(y)
        versions = {"original": y} | {k: fn(y) for k, fn in AUGMENTATIONS.items()}
        for name, sig in versions.items():
            sf.write(AUG_DIR / f"{path.stem}__{name}.wav", sig, SR)
            rows.append({"clip_id": path.stem, "speaker": speaker, "phrase": phrase,
                         "augmentation": name} | extract(sig))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "audio_features.csv", index=False)
    print(f"Saved audio_features.csv: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.groupby("speaker").size().to_string())
    print("\nNaN cells:", int(df.isna().sum().sum()))
    print(df[["clip_id", "augmentation", "duration_s", "rolloff_mean", "energy_mean"]].head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
