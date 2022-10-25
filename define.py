import os


DATA_PATH = "/Users/farina/Workspace/Databases/Docact/acc_user16.csv"
LABEL_PATH = "/Users/farina/Workspace/Databases/Docact/label_user16_edit.csv"
LABEL_COMBINED_PATH = "/Users/farina/Workspace/Databases/Docact/combined_labeled_new.csv"
IMAGE_PATH = "/Users/farina/Workspace/Databases/Docact/image"
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

USER_ID = 16
SCALE_FACTOR = 1.5
SEGMENT_LENGTH = 1  # minute

FEATURE_LEN = 350  # samples
OVERLAP_RATE = 0.5