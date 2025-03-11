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




# Modify ResNet-50 to use GroupNorm instead of BatchNorm
class ResNet50GN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50GN, self).__init__()
        # Load the original ResNet-50 model
        self.resnet50 = models.resnet50(weights=None)

        # Replace the first 7x7 Conv with stride 2 with a 3x3 Conv with stride 1
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the first max pooling layer
        self.resnet50.maxpool = nn.Identity()

        # Replace BatchNorm2d with GroupNorm
        self._replace_batchnorm_with_groupnorm(self.resnet50)

        # Replace the final fully connected layer
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def _replace_batchnorm_with_groupnorm(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                # Replace BatchNorm2d with GroupNorm
                num_channels = module.num_features
                setattr(model, name, nn.GroupNorm(num_groups=8, num_channels=num_channels))
            else:
                # Recursively apply to child modules
                self._replace_batchnorm_with_groupnorm(module)

    def forward(self, x):
        return self.resnet50(x)



if __name__ == "__main__":
    # Initialize the modified ResNet-50
    model = ResNet50GN()
    print(model)