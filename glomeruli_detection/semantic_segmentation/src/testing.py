from torchvision.transforms import functional as F
import torch
import numpy as np
from glomeruli_detection.utils.plot_utils import plot_image_mask_pred
from glomeruli_detection.semantic_segmentation.src.metrics import confusion_matrix

def validate_model(
    model, dataloader, criterion, device=None, with_original=False, plot_interval=20
):
    """
    Validate the model on the given dataloader.

    Args:
        model (torch.nn.Module): The model to be validated.
        dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
        criterion (torch.nn.Module): Loss function.
        device (str, optional): Device to run validation on ('cpu', 'cuda', 'auto', or None).
        with_original (bool, optional): Whether to plot original images.
        plot_interval (int, optional): Interval for plotting images.

    Returns:
        tuple: Validation loss and confusion matrix.
    """
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device is not None:
        device = torch.device(device)
    model.to(device)

    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    conf_matrix = np.zeros(4, dtype=np.uint32)

    with torch.no_grad():
        if with_original:
            for batch_idx, (inputs, labels, original) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device).squeeze(1).float()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # Thresholding the outputs to get binary predictions
                predicted = (outputs > 0.5).float()

                # Plot image, label, and prediction at specified intervals
                if batch_idx % plot_interval == 0:
                    if with_original:
                        plot_image_mask_pred(
                            original[0].permute(1, 2, 0).cpu().numpy(),
                            labels[0].cpu().numpy(),
                            predicted[0].cpu().numpy(),
                        )
                    else:
                        plot_image_mask_pred(
                            inputs[0].permute(1, 2, 0).cpu().numpy(),
                            labels[0].cpu().numpy(),
                            predicted[0].cpu().numpy(),
                        )

                # Update accuracy metrics
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Update confusion matrix
                conf_matrix += confusion_matrix(predicted, labels)
        else:
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device).squeeze(1).float()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # Thresholding the outputs to get binary predictions
                predicted = (outputs > 0.5).float()

                # Plot image, label, and prediction at specified intervals
                if batch_idx % plot_interval == 0:
                    if with_original:
                        plot_image_mask_pred(
                            original[0].permute(1, 2, 0).cpu().numpy(),
                            labels[0].cpu().numpy(),
                            predicted[0].cpu().numpy(),
                        )
                    else:
                        plot_image_mask_pred(
                            inputs[0].permute(1, 2, 0).cpu().numpy(),
                            labels[0].cpu().numpy(),
                            predicted[0].cpu().numpy(),
                        )

                # Update accuracy metrics
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Update confusion matrix
                conf_matrix += confusion_matrix(predicted, labels)

    # Calculate average loss
    val_loss /= len(dataloader.dataset)

    return val_loss, conf_matrix
