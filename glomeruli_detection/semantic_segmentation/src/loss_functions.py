import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice Loss function for measuring overlap between two samples.

    The Dice coefficient is a measure of overlap between two samples:
    - 1 indicates perfect overlap
    - 0 indicates no overlap
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass to compute Dice loss.

        Args:
            inputs (torch.Tensor): Predicted outputs.
            targets (torch.Tensor): Ground truth labels.
            smooth (float, optional): Smoothing factor to avoid division by zero.

        Returns:
            torch.Tensor: Computed Dice loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Dice Loss function.

    This loss function combines the BCE loss and the Dice loss to leverage
    the benefits of both:
    - BCE Loss ensures pixel-wise classification accuracy.
    - Dice Loss ensures overlap and shape similarity.
    """
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        """
        Forward pass to compute combined BCE and Dice loss.

        Args:
            inputs (torch.Tensor): Predicted outputs.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed combined BCE and Dice loss.
        """
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return bce_loss + dice_loss
