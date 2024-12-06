import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
import math
from tqdm import tqdm
from glomeruli_detection.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from glomeruli_detection.utils.wsi_utils import load_wsi


def initialize_data_dir(image_dir, image_patches_dir, image_masks_dir=None):
    """
    Create the directories to save the patches and masks.

    Args:
        image_dir (str): Path to the main directory for image patches and masks.
        image_patches_dir (str): Path to the directory for image patches.
        image_masks_dir (str, optional): Path to the directory for image masks.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(image_patches_dir, exist_ok=True)
    if image_masks_dir is not None:
        os.makedirs(image_masks_dir, exist_ok=True)


def divide_image_masks(
    image_name, patch_size, level, include_empty=False, output_dir=PROCESSED_DATA_DIR
):
    """
    Divide the WSI into patches and generate corresponding masks.

    Args:
        image_name (str): Name of the image file.
        patch_size (int): Size of each patch.
        level (int): Level of the WSI to process.
        include_empty (bool, optional): Whether to include empty patches.
        output_dir (str, optional): Directory to save the processed patches and masks.
    """
    # Paths
    image_dir = os.path.join(output_dir, image_name)
    image_patches_dir = os.path.join(image_dir, "patches")
    image_masks_dir = os.path.join(image_dir, "masks")

    # Initialize the data directories
    initialize_data_dir(image_name, image_patches_dir, image_masks_dir)

    # Load the slide image
    slide_path = os.path.join(RAW_DATA_DIR, image_name + ".svs")
    slide = load_wsi(slide_path)

    # Level information
    level_dim = slide.level_dimensions[level]
    level_downsample = slide.level_downsamples[level]
    level_width, level_height = level_dim
    level_image_shape = (level_height, level_width)

    # Create mask
    annotations, (min_x, min_y, max_x, max_y) = parse_annotations(
        image_name, level_downsample
    )
    mask = create_mask(annotations, level_image_shape)
    assert mask.shape == level_image_shape

    # Patch coordinates
    if include_empty:
        min_x = 0
        min_y = 0
        max_x = level_width
        max_y = level_height
    num_x_patches = math.ceil((max_x - min_x) / patch_size)
    num_y_patches = math.ceil((max_y - min_y) / patch_size)
    num_patches = num_x_patches * num_y_patches
    x_coords = np.linspace(min_x, max_x - patch_size, num_x_patches, dtype=int)
    y_coords = np.linspace(min_y, max_y - patch_size, num_y_patches, dtype=int)

    patch_count = 0
    # Iterate over the slide image to extract patches
    with tqdm(total=num_patches, desc=f"Processing patches for {image_name}") as pbar:
        for y in y_coords:
            for x in x_coords:
                patch_name = f"patch_{x}_{y}"
                patch_filepath = os.path.join(image_patches_dir, patch_name + ".png")
                # Check if the patch already exists
                if os.path.exists(patch_filepath):
                    pbar.update(1)
                    continue

                # Calculate the region to extract
                region = slide.read_region(
                    (
                        math.floor(x * level_downsample),
                        math.floor(y * level_downsample),
                    ),
                    level,
                    (patch_size, patch_size),
                )

                # Convert the region to RGB and remove the alpha channel
                region = region.convert("RGB")

                # Save the mask for the patch
                patch_mask = mask[y : y + patch_size, x : x + patch_size]
                assert patch_mask.shape == (patch_size, patch_size)
                assert np.sum(patch_mask > 0) <= patch_mask.size

                # Check if the mask contains any positive pixels
                if np.any(patch_mask):
                    mask_file_path = os.path.join(image_masks_dir, patch_name + ".npy")
                    np.save(mask_file_path, patch_mask)

                    # Save the patch as an image
                    region.save(patch_filepath)

                # Update the progress bar
                patch_count += 1
                pbar.update(1)

    print(f"Patches saved to {image_patches_dir}")


def divide_image(image_name, patch_size, level, output_dir=PROCESSED_DATA_DIR):
    """
    Divide the WSI into patches without generating masks.

    Args:
        image_name (str): Name of the image file.
        patch_size (int): Size of each patch.
        level (int): Level of the WSI to process.
        output_dir (str, optional): Directory to save the processed patches.
    """
    # Paths
    image_dir = os.path.join(output_dir, image_name)
    image_patches_dir = os.path.join(image_dir, "patches")

    # Initialize the data directories
    initialize_data_dir(image_name, image_patches_dir)

    # Load the slide image
    slide_path = os.path.join(RAW_DATA_DIR, image_name + ".svs")
    slide = load_wsi(slide_path)

    # Level information
    level_dim = slide.level_dimensions[level]
    level_downsample = slide.level_downsamples[level]
    level_width, level_height = level_dim

    # Patch coordinates
    min_x = 0
    min_y = 0
    max_x = level_width
    max_y = level_height
    num_x_patches = math.ceil((max_x - min_x) / patch_size)
    num_y_patches = math.ceil((max_y - min_y) / patch_size)
    num_patches = num_x_patches * num_y_patches
    x_coords = np.linspace(min_x, max_x - patch_size, num_x_patches, dtype=int)
    y_coords = np.linspace(min_y, max_y - patch_size, dtype=int)

    # Iterate over the slide image to extract patches
    with tqdm(total=num_patches, desc=f"Processing patches for {image_name}") as pbar:
        for y in y_coords:
            for x in x_coords:
                patch_name = f"patch_{x}_{y}"
                patch_filepath = os.path.join(image_patches_dir, patch_name + ".png")
                # Check if the patch already exists
                if os.path.exists(patch_filepath):
                    pbar.update(1)
                    continue

                # Calculate the region to extract
                region = slide.read_region(
                    (
                        math.floor(x * level_downsample),
                        math.floor(y * level_downsample),
                    ),
                    level,
                    (patch_size, patch_size),
                )

                # Convert the region to RGB and remove the alpha channel
                region = region.convert("RGB")

                # Save the patch as an image
                region.save(patch_filepath)

                # Update the progress bar
                pbar.update(1)

    print(f"Patches saved to {image_patches_dir}")


def parse_annotations(image_name, factor=1.0):
    """
    Parse the XML file to extract annotation coordinates.

    Args:
        image_name (str): Name of the image file.
        factor (float, optional): Scaling factor for the coordinates.

    Returns:
        tuple: List of annotations and bounding box coordinates (min_x, min_y, max_x, max_y).
    """
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    xml_file_path = os.path.join(RAW_DATA_DIR, f"{image_name}.xml")
    tree = ET.parse(xml_file_path)
    xml_root = tree.getroot()
    annotations = []
    for annotation in xml_root.findall(".//Annotation"):
        coordinates = []
        for coordinate in annotation.find(".//Coordinates"):
            x = float(coordinate.get("X")) / factor
            y = float(coordinate.get("Y")) / factor
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            coordinates.append((x, y))
        annotations.append(coordinates)
    return annotations, (min_x, min_y, max_x, max_y)


def create_mask(annotations, image_shape):
    """
    Create a binary mask from annotation coordinates.

    Args:
        annotations (list): List of annotation coordinates.
        image_shape (tuple): Shape of the mask image.

    Returns:
        np.ndarray: Binary mask.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for coordinates in annotations:
        int_coords = np.array(coordinates, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [int_coords], 255)
    return mask


def process_images(patch_size=2000, level=0):
    """
    Process all images in the RAW_DATA_DIR to create patches and masks.

    Args:
        patch_size (int, optional): Size of each patch.
        level (int, optional): Level of the WSI to process.
    """
    for image_file in os.listdir(RAW_DATA_DIR):
        if image_file.endswith(".svs"):
            image_name = image_file.split(".")[0]
            divide_image_masks(image_name, patch_size, level)


if __name__ == "__main__":
    process_images(patch_size=2000, level=0)
