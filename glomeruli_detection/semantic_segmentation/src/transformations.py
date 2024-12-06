from torchvision.transforms import functional as F
import random
from PIL import Image
from glomeruli_detection.utils.file_utils import load_image, load_mask
from glomeruli_detection.utils.plot_utils import plot_image_mask

def data_augmentation(image, mask, degrees=[0, 90, 270]):
    """
    Apply data augmentation techniques such as rotation and vertical flip.

    Args:
        image (PIL.Image): The input image.
        mask (PIL.Image): The corresponding mask.
        degrees (list, optional): List of degrees to rotate the image and mask.

    Returns:
        tuple: Augmented image and mask.
    """
    angle = random.choice(degrees)
    flip = random.choice([True, False])

    image = F.rotate(image, angle)
    mask = F.rotate(mask, angle)

    if flip:
        image = F.vflip(image)
        mask = F.vflip(mask)

    return image, mask

def resizing(image, mask, size):
    """
    Resize the image and mask to the specified size.

    Args:
        image (PIL.Image): The input image.
        mask (PIL.Image): The corresponding mask (can be None).
        size (tuple): The target size.

    Returns:
        tuple: Resized image and mask if mask is provided, otherwise just the image.
    """
    image = F.resize(image, size)
    if mask is not None:
        mask = F.resize(mask, size)
        return image, mask
    return image

def normalize(image):
    """
    Normalize the image using predefined mean and standard deviation.

    Args:
        image (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    return F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if __name__ == "__main__":
    # Example image and patch names
    image_name, patch_name = "RECHERCHE-004", "patch_10483_60335"

    # Load image and mask
    image = load_image(image_name, patch_name)
    mask = load_mask(image_name, patch_name)

    image = Image.fromarray(image, mode="RGB")
    mask = Image.fromarray(mask, mode="L")

    # Apply resizing
    processed_image, processed_mask = resizing(image, mask, (224, 224))

    # Apply data augmentation
    processed_image, processed_mask = data_augmentation(processed_image, processed_mask)

    # Convert to tensor
    processed_image = F.to_tensor(processed_image)
    processed_mask = F.to_tensor(processed_mask)

    # Apply normalization
    processed_image = normalize(processed_image)

    # Convert back to PIL image for visualization
    processed_image = F.to_pil_image(processed_image)
    processed_mask = F.to_pil_image(processed_mask)

    # Display the images and masks
    plot_image_mask(processed_image, processed_mask)