import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import resnet18


# Hyperparameters
batch_size = 256
temperature = 0.5
learning_rate = 0.001
epochs = 10

# Steps:
# 1 - Data Augmentation
# 2 - Encoder Network
# 3 - Projection Head
# 4 - Contrastive Loss


#          ***         Data Augmentation         ***         #
transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




#          ***         Encoder Network (ResNet-18)         ***         #
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.backbone(x)
    


#          ***         Projection Head         ***         #
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x