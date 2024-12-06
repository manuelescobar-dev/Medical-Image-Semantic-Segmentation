import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def plot_image_mask(image, mask, title=None):
    """
    Plot an image and its mask side by side using Matplotlib.

    Args:
        image (PIL.Image or np.ndarray): Image to plot.
        mask (np.ndarray): Mask to plot.
        title (str, optional): Title for the plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    if title:
        plt.suptitle(title)
    
    plt.show()


def plot_image_mask_pred(image, mask, pred, title=None):
    """
    Plot an image, its mask, and the predicted mask side by side using Matplotlib.

    Args:
        image (PIL.Image or np.ndarray): Image to plot.
        mask (np.ndarray): Mask to plot.
        pred (np.ndarray): Predicted mask to plot.
        title (str, optional): Title for the plot.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Predicted")
    ax[2].axis("off")

    if title:
        plt.suptitle(title)

    plt.show()


def plot_wsi_image_and_mask(image_path, max_dim=10000):
    """
    Assemble and plot the whole slide image (WSI) and its mask from patches.

    Args:
        image_path (str): Path to the directory containing image and mask patches.
        max_dim (int, optional): Maximum dimension to limit the size of the full image and mask.
    """
    patch_dir = os.path.join(image_path, "patches")
    mask_dir = os.path.join(image_path, "masks")

    # Get all patch files
    patch_files = [f for f in os.listdir(patch_dir) if f.endswith(".png")]
    if not patch_files:
        print("No patch files found.")
        return

    # Extract coordinates from patch filenames
    x_coords = []
    y_coords = []
    for patch_file in patch_files:
        _, x, y = os.path.splitext(patch_file)[0].split("_")
        x_coords.append(int(x))
        y_coords.append(int(y))

    # Determine the size of the full image
    patch_img = cv2.imread(os.path.join(patch_dir, patch_files[0]))
    patch_size = patch_img.shape[0]  # assuming square patches
    max_x = max(x_coords) + patch_size
    max_y = max(y_coords) + patch_size

    # Initialize the full image and mask arrays
    full_image = np.zeros((min(max_y, max_dim), min(max_x, max_dim), 3), dtype=np.uint8)
    full_mask = np.zeros((min(max_y, max_dim), min(max_x, max_dim)), dtype=np.uint8)

    # Assemble the full image and mask from patches
    for patch_file in patch_files:
        patch_img = cv2.imread(os.path.join(patch_dir, patch_file))
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)

        _, x, y = os.path.splitext(patch_file)[0].split("_")
        x = int(x)
        y = int(y)

        if x >= max_dim or y >= max_dim:
            continue

        # Calculate the region to be filled
        y_end = min(y + patch_size, max_y, max_dim)
        x_end = min(x + patch_size, max_x, max_dim)
        patch_img_region = patch_img[0 : y_end - y, 0 : x_end - x]

        full_image[y:y_end, x:x_end] = patch_img_region

        mask_file = f"patch_{x}_{y}.npy"
        patch_mask = np.load(os.path.join(mask_dir, mask_file))
        patch_mask_region = patch_mask[0 : y_end - y, 0 : x_end - x]

        full_mask[y:y_end, x:x_end] = patch_mask_region

    # Display the full image and mask
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(full_image)
    ax[0].set_title("Full Image")
    ax[0].axis("off")

    ax[1].imshow(full_mask, cmap="gray")
    ax[1].set_title("Full Mask")
    ax[1].axis("off")

    plt.show()

if __name__ == "__main__":
        # Example usage
    patch_name = "patch_10511_35728"
    image_name = "RECHERCHE-005"

    from glomeruli_detection.utils.file_utils import load_PIL_image_and_mask
    # Load image and mask as PIL Images
    image, mask = load_PIL_image_and_mask(image_name, patch_name)
    
    # Plot the image and mask
    image, mask = plot_image_mask(image, mask )