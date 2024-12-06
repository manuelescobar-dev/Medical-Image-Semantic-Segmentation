from skimage import io
import os
import numpy as np
from PIL import Image
from glomeruli_detection.settings import TRAIN_DATA_DIR


def load_PIL_image_and_mask(image_name, patch_name, data_dir=TRAIN_DATA_DIR):
    """
    Load an image and its corresponding mask as PIL Images.

    Args:
        image_name (str): Name of the image folder (without extension).
        patch_name (str): Name of the patch file (without extension).
        data_dir (str, optional): Base directory where image and mask folders are stored. Defaults to TRAIN_DATA_DIR.

    Returns:
        tuple: A tuple containing:
            - image (PIL.Image.Image): The loaded RGB image.
            - mask (PIL.Image.Image): The loaded mask as a grayscale image.
    """
    # Construct file paths
    image_path = os.path.join(data_dir, "RGB", image_name, patch_name + ".png")
    mask_path = os.path.join(data_dir, "MASK", image_name, patch_name + ".npy")

    # Load image and mask
    image = Image.open(image_path).convert("RGB")
    mask_np = np.load(mask_path)
    mask = Image.fromarray(mask_np).convert("L")

    return image, mask


def load_image(image_name, patch_name, data_dir=TRAIN_DATA_DIR):
    """
    Load an image as a NumPy array.

    Args:
        image_name (str): Name of the image folder (without extension).
        patch_name (str): Name of the patch file (without extension).
        data_dir (str, optional): Base directory where image folders are stored. Defaults to TRAIN_DATA_DIR.

    Returns:
        np.ndarray: The loaded image as a NumPy array in RGB format.
    """
    # Construct file path
    image_path = os.path.join(data_dir, "RGB", image_name, patch_name + ".png")

    # Load image
    image = io.imread(image_path)

    return image


def load_mask(image_name, patch_name, data_dir=TRAIN_DATA_DIR):
    """
    Load a mask as a NumPy array.

    Args:
        image_name (str): Name of the image folder (without extension).
        patch_name (str): Name of the patch file (without extension).
        data_dir (str, optional): Base directory where mask folders are stored. Defaults to TRAIN_DATA_DIR.

    Returns:
        np.ndarray: The loaded mask as a NumPy array.
    """
    # Construct file path
    mask_path = os.path.join(data_dir, "MASK", image_name, patch_name + ".npy")

    # Load mask
    mask_np = np.load(mask_path)

    return mask_np


if __name__ == "__main__":
    # Example usage
    image_name = "IMG1"
    patch_name = "LIT_NOR_VUHSK_127_PAS_VUHSK_20_Aperio_1_1_1"

    # Load image as a NumPy array
    image = load_image(image_name, patch_name)

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Print the shape of the loaded image
    print(f"Image shape: {image_np.shape}")
