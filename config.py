import torch

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# image settings
IMG_SIZE = 112
EMBEDDING_DIM = 128

# training
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 30
WEIGHT_DECAY = 1e-4

# triplet loss margin
MARGIN = 0.5

# data paths
DATA_DIR = 'data/classification_data'
VERIFICATION_DIR = 'data/verification_data'
VERIFICATION_PAIRS = 'data/verification_pairs_val.txt'
EMOTION_DATA_DIR = 'data/emotion_data'
LIVENESS_DATA_DIR = 'data/liveness_data'

# where trained models are saved
MODEL_SAVE_DIR = 'saved_models'

# emotion class labels (FER2013 order)
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NUM_EMOTIONS = 7

# cosine similarity threshold for face verification
VERIFICATION_THRESHOLD = 0.6
