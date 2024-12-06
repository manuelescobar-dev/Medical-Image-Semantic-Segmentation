import os
from glomeruli_detection.settings import RAW_DATA_DIR
from glomeruli_detection.preprocessing.wsi_utils import load_wsi


def image_dimensions(filename):
    """
    Get the dimensions of a whole slide image (WSI).

    Args:
        filename (str): The filename of the WSI (without extension).

    Returns:
        tuple: The dimensions of the WSI.
    """
    slide_path = os.path.join(RAW_DATA_DIR, filename + ".svs")
    slide = load_wsi(slide_path)
    return slide.dimensions


def print_image_info(image_name):
    """
    Print information about a whole slide image (WSI).

    Args:
        image_name (str): The name of the WSI (without extension).
    """
    slide_path = os.path.join(RAW_DATA_DIR, image_name + ".svs")
    slide = load_wsi(slide_path)
    
    print(f"Filename: {image_name}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Levels: {slide.level_count}")
    print(f"Downsamples: {slide.level_downsamples}")
    
    # Print the dimensions at each level
    for i in range(slide.level_count):
        print(f"Dimensions at level {i}: {slide.level_dimensions[i]}")
    print()


if __name__ == "__main__":
    # Iterate through files in RAW_DATA_DIR and print info for each .svs file
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".svs"):
            print_image_info(filename[:-4])
