import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from glomeruli_detection.semantic_segmentation.models.segnet_vgg16 import SegNetVGG16
from glomeruli_detection.semantic_segmentation.src.loss_functions import BCEDiceLoss
from glomeruli_detection.semantic_segmentation.src.loaders import get_dataloaders
from glomeruli_detection.semantic_segmentation.src.training import train_model

if __name__ == "__main__":
    # Hyperparameters
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    step_size = 5
    gamma = 0.1
    num_epochs = 15
    batch_size = 4
    input_size = (224, 224)

    # Get dataloaders for training and validation sets
    dataloaders = get_dataloaders(input_size, batch_size=batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Initialize the model, loss function, optimizer, and learning rate scheduler
    model = SegNetVGG16(num_classes=1)
    criterion = BCEDiceLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
    )
