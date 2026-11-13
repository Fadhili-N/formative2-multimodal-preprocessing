# Multimodal User Authentication & Product Recommendation

**Formative 2 – Multimodal Data Preprocessing** · Machine Learning Pipeline · ALU, 2026 September Term

A command-line system that only gives a product recommendation after it has verified who the user is. A user has to pass **two biometric checks**, face recognition and then voiceprint verification, before the product recommendation is shown. If any check fails, access is denied.

---

## System Flow

```
            ┌──────────────────┐
            │   Face image     │
            └────────┬─────────┘
                     ▼
          ┌─────────────────────┐   unknown face    ┌───────────────┐
          │ Facial Recognition  │ ────────────────▶ │ ACCESS DENIED │
          └──────────┬──────────┘                   └───────────────┘
                     │ recognised member
                     ▼
          ┌─────────────────────┐
          │ Product prediction  │  prepared, not shown yet
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐   unknown voice /  ┌───────────────┐
          │ Voiceprint          │   speaker ≠ face   │ ACCESS DENIED │
          │ Verification        │ ─────────────────▶ └───────────────┘
          └──────────┬──────────┘
                     │ approved voice
                     ▼
          ┌─────────────────────┐
          │ Display recommended │
          │ product             │
          └─────────────────────┘
```

### Multimodal decision rule

A recommendation is shown only if all three conditions hold:

1. **Face accepted.** The face model names a member, and the face's similarity to that member's enrolled photos is above the face threshold.
2. **Voice accepted.** The voice model's confidence is above the voice threshold.
3. **Same person.** The speaker identified by the voice model is the member identified by the face model, so one member's face cannot be combined with another member's voice.

The face and voice models were trained independently on different modalities. Requiring both to agree makes the system much harder to fool than either check alone.

---

## Repository Structure

```
formative2-multimodal-preprocessing/
├── main.py                          # CLI app: face → product → voice → result
├── requirements.txt                 # shared environment
├── README.md
│
├── armstrong_product_model/         # Tabular data + product recommendation model
│   ├── customer_social_profiles.csv
│   ├── customer_transactions.csv
│   ├── merged_dataset.csv           # cleaned, merged, feature-engineered
│   ├── product_model.py
│   └── Armstrong_product_model_notebook.ipynb
│
├── face/                            # Facial images + facial recognition model
│   ├── armstrong/ ayoo/ diane/ fadhili/   # 3 expressions per member
│   ├── processed/                   # aligned faces + augmented copies
│   ├── test_images/                 # unauthorized inputs for the demo
│   ├── models/                      # pretrained YuNet detector + SFace embedder (ONNX)
│   ├── image_features.csv
│   ├── face_model.py
│   ├── face_model.joblib
│   ├── face_metrics.json
│   └── face_recognition.ipynb
│
└── audio/                           # Voice recordings + voiceprint verification model
    ├── raw/                         # original recordings (wav / ogg / mp4)
    ├── converted/                   # 16 kHz mono wav
    ├── unauthorized/                # non-member voices for the demo
    ├── plots/                       # waveforms, spectrograms, confusion matrix
    ├── models/                      # voice_model.joblib, voice_metrics.csv
    ├── audio_features.csv
    ├── audio_pipeline.py            # clean → augment → extract features
    ├── voice_model.py
    └── audio_voice_section.ipynb
```

---

## Data, Preprocessing & Features

### Tabular data (`armstrong_product_model/`)
- **Sources:** `customer_transactions.csv` (150 transactions, 75 customers) and `customer_social_profiles.csv` (155 profile rows, 84 customers).
- **Cleaning:** duplicate rows removed, dates and numeric fields converted to proper types, missing purchase amounts and ratings filled with the median.
- **Key alignment:** `customer_id_legacy` (e.g. `151`) and `customer_id_new` (e.g. `A151`) were unified into one numeric `customer_id`. 61 customers appear in both sources.
- **Join:** social profiles were first aggregated to one row per customer, so the merge cannot duplicate transactions. They were then **left-joined** onto transactions, which keeps every transaction because each one carries the target label `product_category`.
- **Engineered features:** number of prior purchases, average prior purchase amount and average prior rating (computed only from earlier transactions, so no future information leaks in), number of social platforms, most common platform, average engagement, average purchase interest, dominant review sentiment, and engagement level.

### Facial images (`face/`)
- **Data:** 4 members × 3 expressions (neutral, happy, surprised).
- **Preprocessing:** faces are detected and aligned to 112×112 with the pretrained YuNet detector.
- **Augmentations (4 per image):** rotation (±8–15°), horizontal flip, grayscale and brightness/contrast change, giving 60 face samples.
- **Features** (`image_features.csv`): a 128-dimensional SFace embedding (identity) and a 32-bin intensity histogram (lighting).

### Voice recordings (`audio/`)
- **Data:** 4 members × 2 phrases ("Yes, approve", "Confirm transaction").
- **Preprocessing:** every recording is converted to 16 kHz mono, then the stop-button click is cut, silence is trimmed and the peak is normalised.
- **Augmentations (6 per recording):** pitch shift ±1.5 semitones, time stretch ×0.85 / ×1.15 and Gaussian noise at 20 dB and 10 dB SNR, giving 56 samples.
- **Features** (`audio_features.csv`): 13 MFCCs (mean and std), spectral roll-off, RMS energy, spectral centroid, bandwidth and zero-crossing rate.

---

## Models & Evaluation

| Model | Algorithm | Evaluation strategy | Accuracy | F1 (macro) | Log-loss |
|---|---|---|---|---|---|
| Facial recognition | Logistic Regression on SFace embeddings + similarity threshold | Leave-one-expression-out | 1.000 | 1.000 | 0.223 |
| Voiceprint verification | Logistic Regression on MFCC/spectral features + confidence threshold | Leave-one-recording-out | 0.875 | 0.867 | 0.467 |
| Product recommendation | Random Forest (300 trees, class-balanced) | Customer-grouped 75/25 split | 0.200 | 0.190 | pending retrain |

**Preventing leakage.** In every evaluation, all augmented copies of a test sample stay on the test side, and for the product model all transactions of a test customer stay on the test side. Scores therefore reflect performance on unseen inputs, not near-duplicates of training samples.

**Rejecting strangers.** Each biometric model has an acceptance threshold tuned on impostor trials. In those trials each member is left out of training and treated as a stranger, and the voice model also uses non-member recordings.
- Face: threshold 0.41 cosine similarity; 100% of genuine faces accepted and 100% of strangers rejected.
- Voice: confidence threshold stored in `voice_model.joblib`.

---

## Setup

```bash
git clone git@github.com:Fadhili-N/formative2-multimodal-preprocessing.git
cd formative2-multimodal-preprocessing

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`ffmpeg` is bundled through `imageio-ffmpeg`, so it does not need a separate install.

---

## Running the System

### Interactive mode
```bash
python main.py
```
The app asks for a face image and then a voice sample.

### Single transaction
```bash
python main.py --face face/diane/diane_neutral.jpeg --voice audio/converted/diane_yes_approve.wav
```

### Full demo (all scenarios in a row)
```bash
python main.py --demo
```

| Scenario | Face input | Voice input | Outcome |
|---|---|---|---|
| Full transaction | `face/diane/diane_neutral.jpeg` | `audio/converted/diane_yes_approve.wav` | Product recommendation displayed |
| Unauthorized face | `face/test_images/unauthorized_unreadable_face.jpg` | – | Denied at the face step |
| Unauthorized voice | `face/diane/diane_neutral.jpeg` | `audio/unauthorized/stranger_tts1.wav` | Denied at the voice step |
| Face/voice mismatch | `face/diane/diane_neutral.jpeg` | `audio/converted/fadhili_yes_approve.wav` | Denied, speaker does not match face |

**Member-to-customer link.** The team members are not real customers in the dataset, so each member is linked to one real customer profile (`MEMBER_CUSTOMERS` in `main.py`): Armstrong → 187, Ayomide → 114, Fadhili → 101, Diane → 188. These are customers present in both source datasets with the longest purchase histories.

### Retraining the models
```bash
python face/face_model.py train
python audio/audio_pipeline.py        # regenerates augmented audio + audio_features.csv
python audio/voice_model.py train
python armstrong_product_model/product_model.py
```

---

## Team & Contributions

| Member | Role | Main deliverables |
|---|---|---|
| Armstrong | Data & product model | Cleaning, merge and validation, feature engineering, `merged_dataset.csv`, `product_model.py`, product notebook |
| Ayomide | Images & face model | Face alignment, image augmentation, `image_features.csv`, `face_model.py`, face notebook |
| Fadhili | Audio & voice model | Waveforms and spectrograms, audio augmentation, `audio_features.csv`, `voice_model.py`, audio notebook |
| Diane | EDA, CLI & documentation | EDA on the merged dataset, `main.py`, system simulation and demo video, README, final report |

Every member contributed three facial images and two voice recordings.
