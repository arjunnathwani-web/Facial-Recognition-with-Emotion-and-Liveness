# Facial-Recognition-with-Emotion-and-Liveness

## Requirements

```bash
pip install -r requirements.txt
```

## Data Setup

Arrange your datasets like this:

```
data/
  classification_data/
    train_data/<person_id>/*.jpg
    val_data/<person_id>/*.jpg
    test_data/<person_id>/*.jpg
  verification_data/...
  verification_pairs_val.txt
  emotion_data/
    train/<emotion_class>/*.jpg
    val/<emotion_class>/*.jpg
  liveness_data/
    train/real/*.jpg
    train/fake/*.jpg
    val/real/*.jpg
    val/fake/*.jpg
```

For emotion detection, download FER2013 from Kaggle:

https://www.kaggle.com/datasets/msambare/fer2013

If using the folder version with `train/` and `test/` folders:

```bash
python prepare_fer2013.py --source-dir FER-2013 --out data/emotion_data
```

If using the CSV version:

```bash
python prepare_fer2013.py --csv data/fer2013.csv --out data/emotion_data
```

The folder version maps `test/` to `val/` because `train_emotion.py` expects
`data/emotion_data/val`.

For liveness, use LCC FASD:

https://www.kaggle.com/datasets/faber24/lcc-fasd

Prepare it with:

```bash
python prepare_lcc_fasd.py --source-dir "Large Crowdcollected Face Anti-Spoofing Dataset/LCC_FASD" --out data/liveness_data
```

For a quick local smoke test, add `--sample-per-class 100`. Full liveness
training should use the complete prepared folders.

## Training

Run these in order:

```bash
# 1. Classification-based face recognition
python train_classification.py

# 2. Metric learning (triplet loss)
python train_metric.py

# 3. Liveness detection
python train_antispoofing.py

# 4. Emotion detection
python train_emotion.py
```

Trained models are saved to `saved_models/`.

## Evaluation

```bash
python evaluate.py
```

Outputs AUC scores for both models and both similarity metrics. Saves ROC curve to `roc_curves.png`.

## Running the Attendance System

```bash
python main.py
```

- Stand in front of your webcam
- Press **Register Face** to add yourself to the database
- The system will show your name, emotion, and liveness status in real time
- Attendance is logged automatically to `attendance_log.csv`
- Press **View Log** to see the full attendance history

## Innovative Features

### Margin Sensitivity Analysis (Jaspreet Singh - Individual Feature)

Investigates how different triplet loss margin values affect face verification
performance on the validation set. Rather than retraining from scratch at every
margin value, a post-hoc score scaling approach is used to simulate the effect
of different margins on the already-trained metric learning model.

Run with:

```bash
python tune_margin.py
```

Outputs:
- `margin_sensitivity.png` - line plot of AUC vs margin value (0.1 to 1.5)
- `margin_analysis.csv` - full results table

### Live Headcount with Capacity Warning (Adriel - Individual Feature)

Counts all detected faces in each frame and displays a colour-coded badge
directly on the video feed. A green badge indicates the room is within capacity
and a red badge appears when the number of faces exceeds the configured limit
(default: 5). Secondary faces are highlighted with grey bounding boxes. The
headcount and capacity status are also shown in the info panel alongside
identity and emotion.

This feature runs automatically as part of `main.py` - no separate script needed.

## File Overview

| File | Purpose |
|------|---------|
| config.py | Hyperparameters and paths |
| models.py | FaceEmbeddingNet, EmotionNet, LivenessNet, TripletLoss |
| datasets.py | All Dataset classes and transforms |
| train_classification.py | Classification-based training |
| train_metric.py | Triplet loss training |
| train_antispoofing.py | Liveness detection training |
| train_emotion.py | Emotion detection training |
| prepare_fer2013.py | FER2013 dataset folder/CSV preparation |
| prepare_lcc_fasd.py | LCC FASD liveness dataset preparation |
| evaluate.py | ROC/AUC evaluation |
| main.py | Tkinter GUI attendance application (includes live headcount feature) |
| tune_margin.py | Margin sensitivity analysis - individual innovative feature (Jaspreet) |
| margin_sensitivity.png | Output plot from tune_margin.py |
| margin_analysis.csv | Output results table from tune_margin.py |
