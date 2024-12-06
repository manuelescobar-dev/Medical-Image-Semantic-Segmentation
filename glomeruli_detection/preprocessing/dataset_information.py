import numpy as np
import os
from glomeruli_detection.settings import PROCESSED_DATA_DIR

def number_of_samples():
    """
    Calculate and print the number of patches for each WSI and their percentage contribution to the total number of patches.
    """
    patches = []
    for wsi in os.listdir(PROCESSED_DATA_DIR):
        num_patches = len(os.listdir(os.path.join(PROCESSED_DATA_DIR, wsi, "patches")))
        patches.append((wsi, num_patches))

    total = sum([num for _, num in patches])

    for wsi, num_patches in patches:
        percentage = num_patches / total * 100
        print(f"WSI: {wsi}, Patches: {num_patches}, Percentage: {percentage:.2f}%")

def total_number_of_samples():
    """
    Calculate and print the total number of patches across all WSIs.
    """
    total_patches = 0
    for wsi in os.listdir(PROCESSED_DATA_DIR):
        num_patches = len(os.listdir(os.path.join(PROCESSED_DATA_DIR, wsi, "patches")))
        total_patches += num_patches

    print(f"Total number of patches: {total_patches}")

def class_distribution():
    """
    Calculate and print the distribution of the positive class in the masks.
    """
    total_ones = 0
    total_pixels = 0
    for image_name in os.listdir(PROCESSED_DATA_DIR):
        mask_dir = os.path.join(PROCESSED_DATA_DIR, image_name, "masks")
        for mask_file in os.listdir(mask_dir):
            mask = np.load(os.path.join(mask_dir, mask_file))
            total_ones += np.sum(mask > 0)
            total_pixels += mask.size

    print(f"Total ones: {total_ones}")
    print(f"Total pixels: {total_pixels}")
    print(f"Class distribution: {total_ones / total_pixels * 100:.2f}%")

if __name__ == "__main__":
    # Uncomment the desired function to run
    number_of_samples()
    # total_number_of_samples()
    # class_distribution()
