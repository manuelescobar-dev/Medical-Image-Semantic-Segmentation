from skimage import io
import os
import numpy as np
from PIL import Image
from glomeruli_detection.settings import PROCESSED_DATA_DIR

def load_PIL_image_and_mask(image_name, patch_name):
    """
    Load an image and mask as PIL Images from file paths.

    Args:
        image_name (str): Name of the image file (without extension).
        patch_name (str): Name of the patch file (without extension).

    Returns:
        tuple: A tuple containing:
            - image (PIL.Image): The loaded image.
            - mask (PIL.Image): The loaded mask.
    """
    # Construct file paths
    image_path = os.path.join(PROCESSED_DATA_DIR, image_name, "patches", patch_name + ".png")
    mask_path = os.path.join(PROCESSED_DATA_DIR, image_name, "masks", patch_name + ".npy")

    # Load image and mask
    image = Image.open(image_path).convert("RGB")
    mask_np = np.load(mask_path)
    mask = Image.fromarray(mask_np).convert("L")

    return image, mask

def load_image(image_name, patch_name, image_dir=PROCESSED_DATA_DIR):
    """
    Load an image as a NumPy array from a file path.

    Args:
        image_name (str): Name of the image file (without extension).
        patch_name (str): Name of the patch file (without extension).
        image_dir (str, optional): Directory where the images are stored.

    Returns:
        np.ndarray: The loaded image.
    """
    # Construct file path
    image_path = os.path.join(image_dir, image_name, "patches", patch_name + ".png")
    
    # Load image
    image = io.imread(image_path)
    
    return image

def load_mask(image_name, patch_name, image_dir=PROCESSED_DATA_DIR):
    """
    Load a mask as a NumPy array from a file path.

    Args:
        image_name (str): Name of the image file (without extension).
        patch_name (str): Name of the patch file (without extension).
        image_dir (str, optional): Directory where the masks are stored.

    Returns:
        np.ndarray: The loaded mask.
    """
    # Construct file path
    mask_path = os.path.join(image_dir, image_name, "masks", patch_name + ".npy")
    
    # Load mask
    mask_np = np.load(mask_path)
    
    return mask_np

if __name__ == "__main__":
    # Example usage
    patch_name = "patch_10511_35728"
    image_name = "RECHERCHE-005"

    # Load image and mask as PIL Images
    image, mask = load_PIL_image_and_mask(image_name, patch_name)

    # Convert PIL Images to NumPy arrays for further processing
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Print the shapes of the loaded image and mask
    print(f"Image shape: {image_np.shape}")
    print(f"Mask shape: {mask_np.shape}")
