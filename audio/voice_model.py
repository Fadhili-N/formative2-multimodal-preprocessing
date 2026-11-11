"""Step 5: voiceprint verification model.

  python voice_model.py train            # compare models, save the best one
  python voice_model.py verify FILE      # who is speaking? accepted or rejected?

Evaluation = leave-one-recording-out: hold out ONE original recording (and all 6 of its
augmented copies), train on the other 7, test on the held-out original. Repeat for all 8.
Reported: accuracy, macro-F1, log-loss (the "loss" the assignment asks for).
"""
import subprocess
import sys
import warnings
from pathlib import Path

import imageio_ffmpeg
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, log_loss)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_pipeline import SR, clean, extract

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "voice_model.joblib"
META = ["clip_id", "speaker", "phrase", "augmentation"]
DROP = ["duration_s", "energy_total"]          # depend on phrase length, not on the voice

MODELS = {
    "LogisticRegression": lambda: make_pipeline(StandardScaler(),
                                                LogisticRegression(C=1.0, max_iter=5000)),
    "RandomForest": lambda: make_pipeline(StandardScaler(),
                                          RandomForestClassifier(n_estimators=300, random_state=42)),
}


def load_features():
    df = pd.read_csv(ROOT / "audio_features.csv")
    feats = [c for c in df.columns if c not in META + DROP]
    return df, feats


def load_any(path) -> np.ndarray:
    """Read wav/mp3/mp4/ogg/opus -> 16 kHz mono float array (via ffmpeg)."""
    tmp = ROOT / "models" / "_tmp.wav"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
                    "-vn", "-ac", "1", "-ar", str(SR), str(tmp)], check=True)
    y, _ = librosa.load(tmp, sr=SR)
    return y


def clip_vector(path, feats):
    y = clean(load_any(path))
    return pd.DataFrame([extract(y)])[feats]


def cross_validate(df, feats):
    """Leave-one-recording-out. Returns {model: (metrics dict, y_true, y_pred, max_probs)}."""
    classes = sorted(df.speaker.unique())
    out = {}
    for name, make in MODELS.items():
        yt, yp, proba, conf = [], [], [], []
        for clip in sorted(df.clip_id.unique()):
            train, test = df[df.clip_id != clip], df[(df.clip_id == clip) & (df.augmentation == "original")]
            m = make().fit(train[feats], train.speaker)
            p = m.predict_proba(test[feats])
            full = np.zeros((len(test), len(classes)))
            for j, c in enumerate(m.classes_):
                full[:, classes.index(c)] = p[:, j]
            yt += list(test.speaker); yp += [classes[i] for i in full.argmax(1)]
            proba.append(full); conf += list(full.max(1))
        proba = np.vstack(proba)
        out[name] = ({"accuracy": accuracy_score(yt, yp),
                      "f1_macro": f1_score(yt, yp, average="macro"),
                      "log_loss": log_loss(yt, np.clip(proba, 1e-6, 1), labels=classes)},
                     yt, yp, np.array(conf))
    return out


def impostor_scores(df, feats, name):
    """Leave-one-SPEAKER-out: train without a person, see how confident the model is on them."""
    scores = []
    for spk in sorted(df.speaker.unique()):
        m = MODELS[name]().fit(df[df.speaker != spk][feats], df[df.speaker != spk].speaker)
        test = df[(df.speaker == spk) & (df.augmentation == "original")]
        scores += list(m.predict_proba(test[feats]).max(1))
    return np.array(scores)


def pick_threshold(genuine, impostor):
    """Confidence cut-off that best separates members from strangers (balanced accuracy)."""
    best = (0, 0.5)
    for t in np.linspace(0.3, 0.95, 66):
        bal = ((genuine >= t).mean() + (impostor < t).mean()) / 2
        if bal > best[0] + 1e-9:
            best = (bal, t)
    return best[1], best[0]


def train():
    df, feats = load_features()
    results = cross_validate(df, feats)
    rows = [{"model": n, **r[0]} for n, r in results.items()]
    metrics = pd.DataFrame(rows).round(3)
    metrics.to_csv(ROOT / "models" / "voice_metrics.csv", index=False)
    print("Leave-one-recording-out results (8 test recordings):\n", metrics.to_string(index=False))

    best = metrics.sort_values(["f1_macro", "log_loss"], ascending=[False, True]).iloc[0].model
    print("\nBest model:", best)
    _, yt, yp, genuine = results[best]
    classes = sorted(df.speaker.unique())
    ConfusionMatrixDisplay(confusion_matrix(yt, yp, labels=classes), display_labels=classes).plot(cmap="Blues")
    plt.title(f"{best}: held-out recordings"); plt.savefig(ROOT / "plots" / "voice_confusion_matrix.png", dpi=110)
    plt.close()

    # impostors: members left out of training + non-member voices in unauthorized/
    imp = list(impostor_scores(df, feats, best))
    final = MODELS[best]().fit(df[feats], df.speaker)
    stranger_files = sorted((ROOT / "unauthorized").glob("*")) if (ROOT / "unauthorized").exists() else []
    for f in stranger_files:
        imp.append(final.predict_proba(clip_vector(f, feats)).max())
    imp = np.array(imp)
    thr, bal = pick_threshold(genuine, imp)
    print(f"\nGenuine confidence  mean {genuine.mean():.2f} (n={len(genuine)})")
    print(f"Impostor confidence mean {imp.mean():.2f} (n={len(imp)}, {len(stranger_files)} are non-member voices)")
    print(f"Chosen accept threshold {thr:.2f} -> balanced accuracy {bal:.2f}, "
          f"false-reject {np.mean(genuine < thr):.0%}, false-accept {np.mean(imp >= thr):.0%}")

    joblib.dump({"model": final, "features": feats, "classes": list(final.classes_),
                 "threshold": float(thr), "name": best}, MODEL_PATH)
    print("Saved", MODEL_PATH)


def verify(path):
    bundle = joblib.load(MODEL_PATH)
    p = bundle["model"].predict_proba(clip_vector(path, bundle["features"]))[0]
    who, conf = bundle["classes"][int(p.argmax())], float(p.max())
    ok = conf >= bundle["threshold"]
    return ok, who, conf


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 3 and sys.argv[1] == "verify":
        ok, who, conf = verify(sys.argv[2])
        print(f"{'ACCEPTED' if ok else 'REJECTED'}: closest voice = {who}, confidence {conf:.2f}")
    else:
        print(__doc__)
