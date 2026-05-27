# Face Training and Evaluation Summary

## Face Classification

- Status: Training reached epoch 27/30 before a Windows DataLoader worker crash.
- Best saved validation accuracy: 0.0092.
- Saved model filename used locally: `saved_models/face_classification.pth`.
- Metadata filename used locally: `saved_models/classification_meta.json`.

## Metric Learning

- Status: Training completed.
- Best triplet loss: 0.0873.
- Saved model filename used locally: `saved_models/face_metric.pth`.

## Face Verification Evaluation

- Classification embedding, cosine similarity AUC: 0.8543.
- Classification embedding, Euclidean distance AUC: 0.8543.
- Metric learning embedding, cosine similarity AUC: 0.8449.
- Metric learning embedding, Euclidean distance AUC: 0.8449.
- ROC curve image: `roc_curves.png`.

## Emotion and Liveness

- Emotion detection best validation accuracy: 0.6213.
- Liveness detection best validation accuracy: 0.9695.
- Full logs: `emotion_train.log` and `liveness_train.log`.
