import os

# Train
TRAIN_DATA_DIR = os.path.join("data", "DATASET_A_DIB")
TRAIN_WSI_DATA_DIR = os.path.join("data", "DATASET_A_DIB", "WSI")
TRAIN_RGB_DATA_DIR = os.path.join("data", "DATASET_A_DIB", "RGB")
TRAIN_MASK_DATA_DIR = os.path.join("data", "DATASET_A_DIB", "MASK")

# Test
TEST_DATA_DIR = os.path.join("data", "DATASET_B_DIB")
TEST_WSI_DATA_DIR = os.path.join("data", "DATASET_B_DIB", "WSI")
TEST_RGB_DATA_DIR = os.path.join("data", "DATASET_B_DIB", "RGB")
TEST_MASK_DATA_DIR = os.path.join("data", "DATASET_B_DIB", "MASK")
