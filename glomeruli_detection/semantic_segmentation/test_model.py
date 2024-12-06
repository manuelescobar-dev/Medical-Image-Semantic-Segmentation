import torch
import os
from torch.utils.data import DataLoader

from glomeruli_detection.semantic_segmentation.src.testing import validate_model
from glomeruli_detection.semantic_segmentation.src.loss_functions import BCEDiceLoss
from glomeruli_detection.semantic_segmentation.models.segnet_vgg16 import SegNetVGG16
from glomeruli_detection.settings import MODELS_DIR, TEST_WSI
from glomeruli_detection.semantic_segmentation.src.metrics import evaluate
from glomeruli_detection.semantic_segmentation.src.loaders import GlomeruliDataset

if __name__ == "__main__":
    # Initialize the test dataset
    test_dataset = GlomeruliDataset(
        mode="test",
        wsi_list=TEST_WSI,
        input_size=(224, 224),
        return_original_image=True,
    )
    
    # Create the dataloader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Initialize the model
    model = SegNetVGG16(num_classes=1)

    # Load the pre-trained model weights
    model_path = os.path.join(MODELS_DIR, "SegNetVGG16_1.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model.load_state_dict(torch.load(model_path))

    # Validate the model
    val_loss, conf_matrix = validate_model(
        model, test_loader, BCEDiceLoss(), device="auto", with_original=True, plot_interval=500
    )

    # Print evaluation metrics
    evaluate(conf_matrix)
