import torch
import logging
import os
import numpy as np
from glomeruli_detection.utils.plot_utils import plot_image_mask_pred
from glomeruli_detection.semantic_segmentation.src.testing import validate_model
from glomeruli_detection.settings import MODELS_DIR
from glomeruli_detection.semantic_segmentation.src.metrics import (
    accuracy,
    recall,
    confusion_matrix,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_model_id(model_name):
    """
    Get the next model ID based on existing model files.

    Args:
        model_name (str): Base name of the model.

    Returns:
        int: The next available model ID.
    """
    max_id = 0
    for filename in os.listdir(MODELS_DIR):
        parts = filename.split("_")
        if parts[0] == model_name:
            model_id = int(parts[1].split(".")[0])
            max_id = max(max_id, model_id)
    return max_id + 1

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device="auto",
):
    """
    Train the model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): Dataloader for training data.
        val_loader (torch.utils.data.DataLoader): Dataloader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (str, optional): Device to run training on ('cpu', 'cuda', 'auto', or None).
    """
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device is not None:
        device = torch.device(device)
    model.to(device)

    if device.type == "cuda":
        # Clear cache
        torch.cuda.empty_cache()

    # Get a unique model ID for saving
    model_id = get_model_id(model.__class__.__name__)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        train_conf_matrix = np.zeros(4, dtype=np.uint32)
        predicted_ones = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze(1).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)  # Ensure outputs shape matches labels shape

            assert outputs.shape == labels.shape
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # Accumulate total loss

            predicted = (outputs > 0.5).float()  # Assuming binary classification

            # Plot image, label, and prediction at specified intervals
            if batch_idx % 100 == 0:
                plot_image_mask_pred(
                    inputs[0].permute(1, 2, 0).cpu().numpy(),
                    labels[0].cpu().numpy(),
                    predicted[0].cpu().numpy(),
                )

            # Number of ones in the predicted mask
            predicted_ones += torch.sum(predicted).item()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_conf_matrix += confusion_matrix(predicted, labels)

            if batch_idx % 10 == 0:  # Log every 10 batches
                batch_tpr = recall(train_conf_matrix)
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}, "
                    f"TPR: {batch_tpr:.4f}, Predicted Ones: {predicted_ones/(batch_idx+1):.4f}"
                )

        epoch_loss = running_loss / len(train_loader.dataset)  # Average loss for the epoch

        train_accuracy = accuracy(train_conf_matrix)
        train_tpr = recall(train_conf_matrix)

        val_loss, conf_matrix = validate_model(model, val_loader, criterion, device)
        val_accuracy = accuracy(conf_matrix)
        val_tpr = recall(conf_matrix)
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, TPR: {train_tpr:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val TPR: {val_tpr:.4f}"
        )

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(MODELS_DIR, f"{model.__class__.__name__}_{model_id}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")

    logger.info("Training complete")
