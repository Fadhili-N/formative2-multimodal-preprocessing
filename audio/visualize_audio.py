"""Step 3: waveform + spectrogram for every member/phrase. Saves PNGs in plots/."""
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
WAV, PLOTS = ROOT / "converted", ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
SR = 16000
members = sorted({f.stem.split("_")[0] for f in WAV.glob("*.wav")})
phrases = ["yes_approve", "confirm_transaction"]

# one figure per member: 2 phrases x (waveform, spectrogram)
for m in members:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for col, ph in enumerate(phrases):
        y, sr = librosa.load(WAV / f"{m}_{ph}.wav", sr=SR)
        librosa.display.waveshow(y, sr=sr, ax=axes[0, col], color="#2a6f97")
        axes[0, col].set(title=f"{m} - {ph} (waveform)", xlabel="Time (s)", ylabel="Amplitude")
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
        img = librosa.display.specshow(S, sr=sr, hop_length=128, x_axis="time", y_axis="hz",
                                       ax=axes[1, col], cmap="magma")
        axes[1, col].set(title=f"{m} - {ph} (spectrogram)", ylabel="Frequency (Hz)")
        fig.colorbar(img, ax=axes[1, col], format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(PLOTS / f"{m}_waveform_spectrogram.png", dpi=110)
    plt.close(fig)

# summary numbers to base the written interpretation on
print(f"{'file':34s}{'sec':>6s}{'rms':>8s}{'speech%':>9s}{'lead_sil':>9s}{'clip%':>7s}{'cent_Hz':>8s}{'f0_Hz':>7s}")
for m in members:
    for ph in phrases:
        y, sr = librosa.load(WAV / f"{m}_{ph}.wav", sr=SR)
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
        voiced = rms > 0.1 * rms.max()
        lead = np.argmax(voiced) * 128 / sr
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=1024)
        f0 = np.median(f0[np.repeat(voiced, 1)[: len(f0)]]) if voiced.any() else np.nan
        cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        print(f"{m + '_' + ph:34s}{len(y)/sr:6.2f}{np.sqrt((y**2).mean()):8.3f}{voiced.mean()*100:9.0f}"
              f"{lead:9.2f}{(np.abs(y) > 0.99).mean()*100:7.2f}{cent:8.0f}{f0:7.0f}")
