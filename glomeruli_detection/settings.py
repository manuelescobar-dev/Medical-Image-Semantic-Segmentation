import os

# Path to the root directory of semantic segmentation
SEMANTIC_SEGMENTATION_DIR = os.path.join("glomeruli_detection", "semantic_segmentation")

# Path to the raw data directory containing the wsi and xml files
RAW_DATA_DIR = os.path.join("data", "raw")

# Path to the processed data directory containing the patches
PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Path to the directory containing the glomeruli RGB images
GL_IMAGE_DIR = os.path.join(PROCESSED_DATA_DIR, "glomeruli_rgb")

# Path to the models directory containing the trained models
MODELS_DIR = os.path.join(SEMANTIC_SEGMENTATION_DIR, "outputs")

# Path to the directory containing the openslide library, downloaded from https://openslide.org/download/
OPENSLIDE_DIR = "D:/MLA/mla-prj-24-mla24_prfp04_g06/data/openslide"

# Train, validation and test WSI split
TRAIN_WSI = [
    "RECHERCHE-003",
    "RECHERCHE-009",
    "RECHERCHE-010",
    "RECHERCHE-011",
    "RECHERCHE-015",
    "RECHERCHE-016",
    "RECHERCHE-017",
]
VAL_WSI = [
    "RECHERCHE-004",
]
TEST_WSI = [
    "RECHERCHE-005",
]

# Path to the pipeline output directory
PIPELINE_OUTPUT_DIR = os.path.join("glomeruli_detection", "pipeline", "outputs")
