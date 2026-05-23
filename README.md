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

For emotion detection, download FER2013 from Kaggle and restructure it into the above folders.
For liveness, use LCC FASD or CelebA-Spoof datasets.

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
| evaluate.py | ROC/AUC evaluation |
| main.py | Tkinter GUI attendance application |
