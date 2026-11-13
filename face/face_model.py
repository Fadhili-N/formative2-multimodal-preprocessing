"""Facial recognition model (step 1 of the authentication system).

Steps:
    1. Find the face in each photo and align it          (pretrained YuNet + SFace)
    2. Make 4 augmented copies: rotate, flip, grayscale, brightness
    3. Turn each face into 128 numbers (an "embedding") -> image_features.csv
    4. Train Logistic Regression to name the member
    5. Reject strangers: the face must also look similar enough to that
       member's enrolled photos

Usage:
    python face_model.py train
    python face_model.py predict path/to/photo.jpg
"""

import json
import os
import sys
from pathlib import Path
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide noisy OpenCV warnings
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MEMBERS = ["armstrong", "ayoo", "diane", "fadhili"]
EXPRESSIONS = ["neutral", "happy", "surprised"]
AUGMENTATIONS = ["original", "rotate", "flip", "grayscale", "brightness"]
EMB_COLS = [f"emb_{i}" for i in range(128)]  # embedding features (used by the model)
HIST_COLS = [f"hist_{i}" for i in range(32)]  # histogram features (saved, not used by the model)
SEED = 42

MODEL_PATH = ROOT / "face_model.joblib"

# Two pretrained networks from the OpenCV model zoo (we do not train them).
detector = cv2.FaceDetectorYN.create(str(ROOT / "models/face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6)
recognizer = cv2.FaceRecognizerSF.create(str(ROOT / "models/face_recognition_sface_2021dec.onnx"), "")


# 1. Load photos and find faces
def list_photos():
    """Table of all photos: source_id (path), member, expression."""
    rows = []
    for member in MEMBERS:
        for path in sorted((ROOT / member).glob(f"{member}_*.jp*g")):
            expression = path.stem.split("_")[1]
            rows.append({"source_id": f"{member}/{path.name}", "member": member, "expression": expression})
    return pd.DataFrame(rows)


def load_image(path):
    """Read a photo (BGR) and shrink it so the longest side is at most 1024 px."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    scale = 1024 / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def detect_face(image):
    """Return (aligned 112x112 face, detection box), or None if there is no face."""
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None:
        return None
    best = faces[faces[:, -1].argmax()]  # the last column is the confidence score
    return recognizer.alignCrop(image, best), best


# 2. Augmentation
def augment(face, seed):
    """The original face plus 4 augmented copies."""
    rng = np.random.default_rng(seed)
    h, w = face.shape[:2]
    angle = rng.choice([-1, 1]) * rng.uniform(8, 15)
    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return {
        "original": face,
        "rotate": cv2.warpAffine(face, rotation, (w, h), borderMode=cv2.BORDER_REFLECT),
        "flip": cv2.flip(face, 1),
        "grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "brightness": cv2.convertScaleAbs(face, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-30, 30)),
    }


# 3. Features
def embed(face):
    """128 numbers describing who the face belongs to (length 1, so dot product = similarity)."""
    e = recognizer.feature(face).ravel()
    return e / np.linalg.norm(e)


def histogram(face):
    """32-bin brightness histogram (describes lighting, not identity)."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    return h / h.sum()


def build_features(save_images=True):
    """One row per photo x augmentation (12 x 5 = 60 rows)."""
    rows = []
    for i, photo in list_photos().iterrows():
        face, _ = detect_face(load_image(ROOT / photo.source_id))
        for name, view in augment(face, seed=SEED + i).items():
            if save_images:
                out = ROOT / "processed" / photo.member / f"{Path(photo.source_id).stem}_{name}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), view)
            rows.append([photo.source_id, photo.member, photo.expression, name, *embed(view), *histogram(view)])
    return pd.DataFrame(rows, columns=["source_id", "member", "expression", "augmentation", *EMB_COLS, *HIST_COLS])

# 4. Model and evaluation
def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


def similarity_to_member(embedding, gallery, member):
    """Highest similarity between a face and any enrolled face of `member`."""
    return (gallery.loc[gallery.member == member, EMB_COLS].to_numpy() @ embedding).max()


def evaluate(features, make=make_model):
    """Leave-one-expression-out: train on 2 expressions, test on the original photos of the 3rd.

    This way no augmented copy of a test photo is ever used for training.
    """
    results, probabilities = [], []
    for held_out in EXPRESSIONS:
        train = features[features.expression != held_out]
        test = features[(features.expression == held_out) & (features.augmentation == "original")]
        model = make().fit(train[EMB_COLS].to_numpy(), train.member)
        proba = model.predict_proba(test[EMB_COLS].to_numpy())
        probabilities.append(proba)
        for (_, row), p in zip(test.iterrows(), proba):
            guess = model.classes_[p.argmax()]
            results.append({"source_id": row.source_id, "actual": row.member, "predicted": guess,
                            "probability": p.max(),
                            "similarity": similarity_to_member(row[EMB_COLS].to_numpy(float), train, guess)})
    results = pd.DataFrame(results)
    scores = {
        "accuracy": accuracy_score(results.actual, results.predicted),
        "macro_f1": f1_score(results.actual, results.predicted, average="macro"),
        "log_loss": log_loss(results.actual, np.vstack(probabilities), labels=MEMBERS),
        "confusion_matrix": confusion_matrix(results.actual, results.predicted, labels=MEMBERS).tolist(),
    }
    return results, scores


def test_impostors(features):
    """Leave-one-member-out: train without one member, then show the model that member.

    That member is a stranger to the model, so this simulates unauthorized people.
    """
    results = []
    for stranger in MEMBERS:
        train = features[features.member != stranger]
        test = features[(features.member == stranger) & (features.augmentation == "original")]
        model = make_model().fit(train[EMB_COLS].to_numpy(), train.member)
        for (_, row), p in zip(test.iterrows(), model.predict_proba(test[EMB_COLS].to_numpy())):
            guess = model.classes_[p.argmax()]
            results.append({"source_id": row.source_id, "stranger": stranger, "predicted": guess,
                            "probability": p.max(),
                            "similarity": similarity_to_member(row[EMB_COLS].to_numpy(float), train, guess)})
    return pd.DataFrame(results)


def train():
    """Build features, evaluate, choose the threshold, train the final model, and save everything."""
    features = build_features()
    features.to_csv(ROOT / "image_features.csv", index=False)

    predictions, scores = evaluate(features)
    impostors = test_impostors(features)

    # Threshold: halfway between the most similar stranger and the least similar real member.
    threshold = (impostors.similarity.max() + predictions.similarity.min()) / 2

    model = make_model().fit(features[EMB_COLS].to_numpy(), features.member)
    gallery = features[features.augmentation == "original"]  # enrolled faces used for the similarity check
    joblib.dump({"model": model, "gallery": gallery, "threshold": threshold}, MODEL_PATH)

    metrics = {**scores, "threshold": threshold,
               "genuine_accepted": float((predictions.similarity >= threshold).mean()),
               "strangers_rejected": float((impostors.similarity < threshold).mean())}
    (ROOT / "face_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"features": features, "metrics": metrics, "predictions": predictions, "impostors": impostors}


# 5. Prediction (used by the CLI app)
def predict(image_path, saved=None):
    """Check one photo. Returns authorized, member, probability, similarity, reason."""
    saved = saved or joblib.load(MODEL_PATH)
    found = detect_face(load_image(image_path))
    if found is None:
        return {"authorized": False, "member": None, "probability": 0.0, "similarity": 0.0,
                "reason": "no face detected"}

    embedding = embed(found[0])
    proba = saved["model"].predict_proba(embedding[None, :])[0]
    guess = saved["model"].classes_[proba.argmax()]
    similarity = similarity_to_member(embedding, saved["gallery"], guess)
    authorized = bool(similarity >= saved["threshold"])
    return {"authorized": authorized, "member": guess if authorized else None,
            "probability": float(proba.max()), "similarity": float(similarity),
            "reason": f"matches {guess}" if authorized else f"too different from {guess} ({similarity:.2f})"}


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        m = train()["metrics"]
        print(f"accuracy {m['accuracy']:.3f} | macro F1 {m['macro_f1']:.3f} | log loss {m['log_loss']:.3f}")
        print(f"threshold {m['threshold']:.3f} | genuine accepted {m['genuine_accepted']:.0%} | "
              f"strangers rejected {m['strangers_rejected']:.0%}")
    elif len(sys.argv) == 3 and sys.argv[1] == "predict":
        r = predict(sys.argv[2])
        print(("ACCESS GRANTED: " + r["member"]) if r["authorized"] else "ACCESS DENIED", f"({r['reason']})")
    else:
        print(__doc__)
