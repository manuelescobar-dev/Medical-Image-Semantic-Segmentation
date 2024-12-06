import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import copy

from glomeruli_detection.semantic_segmentation.src.transformations import (
    data_augmentation,
    normalize,
    resizing,
)
from glomeruli_detection.settings import (
    PROCESSED_DATA_DIR,
    TEST_WSI,
    TRAIN_WSI,
    VAL_WSI,
)
from glomeruli_detection.utils.file_utils import load_image, load_mask


class GlomeruliDataset(Dataset):
    def __init__(
        self,
        mode,
        wsi_list,
        input_size,
        return_patch_name=False,
        return_original_image=False,
        image_dir=PROCESSED_DATA_DIR,
    ):
        """
        Initialize the GlomeruliDataset.

        Args:
            mode (str): The mode of the dataset ('train', 'val', 'test', or 'production').
            wsi_list (list): List of whole slide images (WSI).
            input_size (tuple): The target size for the images.
            return_patch_name (bool, optional): Whether to return the patch name.
            return_original_image (bool, optional): Whether to return the original image.
            image_dir (str, optional): Directory where the processed data is stored.
        """
        self.mode = mode
        self.wsi_list = wsi_list
        self.return_patch_name = return_patch_name
        self.return_original_image = return_original_image
        self.input_size = input_size
        self.image_dir = image_dir

        self.patch_list = []
        for image_name in self.wsi_list:
            image_path = os.path.join(image_dir, image_name)
            patches_path = os.path.join(image_path, "patches")

            for patch_filename in os.listdir(patches_path):
                patch_name = patch_filename.split(".")[0]
                self.patch_list.append((image_name, patch_name))

    def __len__(self):
        """
        Return the number of patches in the dataset.
        """
        return len(self.patch_list)

    def __getitem__(self, idx):
        """
        Retrieve a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: The image, mask (if applicable), and optionally patch name and original image.
        """
        image_name, patch_name = self.patch_list[idx]

        # Load image
        image = load_image(image_name, patch_name, self.image_dir)
        image = Image.fromarray(image, mode="RGB")

        if self.mode != "production":
            mask = load_mask(image_name, patch_name, self.image_dir)
            mask = Image.fromarray(mask, mode="L")

        # Apply resizing
        if self.mode == "production":
            image = resizing(image, None, self.input_size)
        else:
            image, mask = resizing(image, mask, self.input_size)

        # Apply data augmentation if in training mode
        if self.mode == "train":
            image, mask = data_augmentation(image, mask)

        image = F.to_tensor(image)
        if self.mode != "production":
            mask = F.to_tensor(mask)

        if self.return_original_image:
            original_image = copy.deepcopy(image)

        # Apply normalization
        image = normalize(image)

        # Return the appropriate set of items based on the mode and flags
        if self.mode == "production":
            if self.return_patch_name:
                return image, patch_name
            return image
        else:
            if self.return_patch_name and self.return_original_image:
                return image, mask, patch_name, original_image
            elif self.return_patch_name:
                return image, mask, patch_name
            elif self.return_original_image:
                return image, mask, original_image
            else:
                return image, mask


def get_dataloaders(input_size=(256, 256), batch_size=4):
    """
    Load data and create dataloaders for training, validation, and test sets.

    Args:
        input_size (tuple, optional): The target size for the images.
        batch_size (int, optional): Batch size for the dataloaders.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' dataloaders.
    """
    # Train dataset and dataloader
    train_dataset = GlomeruliDataset(
        mode="train", wsi_list=TRAIN_WSI, input_size=input_size
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Validation dataset and dataloader
    val_dataset = GlomeruliDataset(
        mode="val", wsi_list=VAL_WSI, input_size=input_size
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Test dataset and dataloader
    test_dataset = GlomeruliDataset(
        mode="test", wsi_list=TEST_WSI, input_size=input_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Return the dataloaders in a dictionary
    return {"train": train_loader, "val": val_loader, "test": test_loader}