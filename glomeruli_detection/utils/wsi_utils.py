import os
import openslide
from glomeruli_detection.settings import TRAIN_WSI_DATA_DIR


def image_dimensions(path, image_name):
    """
    Get the dimensions of a whole slide image (WSI).

    Args:
        path (str): The directory containing the WSI file.
        image_name (str): The filename of the WSI (including extension).

    Returns:
        tuple: The dimensions of the WSI as (width, height).
    """
    slide_path = os.path.join(path, image_name)
    slide = load_wsi(slide_path)
    return slide.dimensions


def print_image_info(path, image_name):
    """
    Print information about a whole slide image (WSI).

    Args:
        path (str): The directory containing the WSI file.
        image_name (str): The filename of the WSI (including extension).

    Prints:
        - Filename of the WSI.
        - Dimensions of the WSI.
        - Number of pyramid levels in the WSI.
        - Downsampling factors for each pyramid level.
        - Dimensions at each pyramid level.
    """
    slide_path = os.path.join(path, image_name)
    slide = load_wsi(slide_path)

    print(f"Filename: {image_name}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Levels: {slide.level_count}")
    print(f"Downsamples: {slide.level_downsamples}")

    # Print the dimensions at each level
    for i in range(slide.level_count):
        print(f"Dimensions at level {i}: {slide.level_dimensions[i]}")
    print()


def load_wsi(slide_path):
    """
    Load a whole slide image (WSI) using OpenSlide.

    Args:
        slide_path (str): The full file path to the WSI.

    Returns:
        openslide.OpenSlide: An OpenSlide object representing the WSI.

    Raises:
        openslide.OpenSlideError: If the file cannot be opened as a WSI.
    """
    slide = openslide.OpenSlide(slide_path)
    return slide


if __name__ == "__main__":
    # Iterate through files in TRAIN_WSI_DATA_DIR and print info for each .svs file
    for filename in os.listdir(TRAIN_WSI_DATA_DIR):
        if filename.endswith(".svs"):
            print_image_info(TRAIN_WSI_DATA_DIR, filename)
