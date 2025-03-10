import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet50(nn.Module):
    """
    Replaces the first 7x7 convolution with stride 2 with a 3x3 convolution with stride 1.
    And removes the first max pooling operation.
    These changes ensure that the network can effectively process the smaller CIFAR-10 images 
    without losing too much spatial information early in the network.
    """
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        # Load the original ResNet-50 model
        self.resnet50 = models.resnet50(weights=None)

        # Replace the first 7x7 Conv with stride 2 with a 3x3 Conv with stride 1
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the first max pooling layer
        self.resnet50.maxpool = nn.Identity()

    def forward(self, x):
        return self.resnet50(x)


if __name__ == "__main__":
    # Initialize the modified ResNet-50
    model = ModifiedResNet50()
    print(model)