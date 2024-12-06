import os
from glomeruli_detection.settings import OPENSLIDE_DIR

# Ensure the OpenSlide library is loaded correctly by adding its directory to the DLL search path
with os.add_dll_directory(os.path.join(OPENSLIDE_DIR, "bin")):
    import openslide

def load_wsi(slide_path):
    """
    Load a whole slide image (WSI) using OpenSlide.

    Args:
        slide_path (str): The file path to the WSI.

    Returns:
        openslide.OpenSlide: An OpenSlide object representing the WSI.
    """
    slide = openslide.OpenSlide(slide_path)
    return slide
