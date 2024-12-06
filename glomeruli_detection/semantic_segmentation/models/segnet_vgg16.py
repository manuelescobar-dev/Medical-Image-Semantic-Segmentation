import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class SegNetVGG16(nn.Module):
    def __init__(self, num_classes):
        super(SegNetVGG16, self).__init__()

        # Load pretrained VGG16 with batch normalization
        vgg16 = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        features = list(vgg16.features.children())

        # Encoder layers from VGG16
        self.enc1 = nn.Sequential(*features[0:6])  # Conv2d(3, 64) to ReLU
        self.enc2 = nn.Sequential(*features[7:13])  # Conv2d(64, 128) to ReLU
        self.enc3 = nn.Sequential(*features[14:23])  # Conv2d(128, 256) to ReLU
        self.enc4 = nn.Sequential(*features[24:33])  # Conv2d(256, 512) to ReLU
        self.enc5 = nn.Sequential(*features[34:43])  # Conv2d(512, 512) to ReLU

        # Decoder layers
        self.dec5 = self._make_decoder_layer3(512, 512, 512)
        self.dec4 = self._make_decoder_layer3(512, 512, 256)
        self.dec3 = self._make_decoder_layer3(256, 256, 128)
        self.dec2 = self._make_decoder_layer2(128, 64)
        self.dec1 = self._make_decoder_layer2(64, 64)

        # Final convolution layer to get desired number of classes
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    # Name of the model
    def __str__(self):
        return "SegNetVGG16"

    def forward(self, x):
        # Encoder with max pooling
        x1, ind1 = self._max_pool(self.enc1(x))
        x2, ind2 = self._max_pool(self.enc2(x1))
        x3, ind3 = self._max_pool(self.enc3(x2))
        x4, ind4 = self._max_pool(self.enc4(x3))
        x5, ind5 = self._max_pool(self.enc5(x4))

        # Decoder with max unpooling
        x5d = self._max_unpool(x5, ind5, output_size=x4.size())
        x5d = self.dec5(x5d)
        x4d = self._max_unpool(x5d, ind4, output_size=x3.size())
        x4d = self.dec4(x4d)
        x3d = self._max_unpool(x4d, ind3, output_size=x2.size())
        x3d = self.dec3(x3d)
        x2d = self._max_unpool(x3d, ind2, output_size=x1.size())
        x2d = self.dec2(x2d)
        x1d = self._max_unpool(x2d, ind1, output_size=x.size())
        x1d = self.dec1(x1d)

        # Final output layer
        out = self.final_conv(x1d)

        # Sigmoid activation function
        out = self.sigmoid(out)

        return out

    def _make_decoder_layer3(self, in_channels, mid_channels, out_channels):
        """Helper function to create a decoder layer."""
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_decoder_layer2(self, in_channels, out_channels):
        """Helper function to create a decoder layer."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _max_pool(self, x):
        """Max pooling with indices."""
        pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        return pool(x)

    def _max_unpool(self, x, indices, output_size):
        """Max unpooling with output size."""
        unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        return unpool(x, indices=indices, output_size=output_size)
