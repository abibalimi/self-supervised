import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import time


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
        self.backbone = resnet18(weights=None)   # Random initialization
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
    
    
    


#          ***         SimCLR Model         ***         #
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_head):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x1, x2):
        # Encode the two augmented views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # Project to the lower-dimensional space
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        return z1, z2

    
    
    
#          ***         Contrastive Loss (NT-Xent)         ***         #
def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # Concatenate both views
    z = nn.functional.normalize(z, dim=1)  # Normalize feature vectors

    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature

    # Create labels for positive pairs
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])  # Positive pairs are diagonal elements

    # Compute cross-entropy loss
    loss = nn.functional.cross_entropy(sim_matrix, labels)
    return loss





# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")





# Initialize model, optimizer, and loss
encoder = Encoder()
projection_head = ProjectionHead()
model = SimCLR(encoder, projection_head)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)





# Training loop
t0 = time.time()
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        # Generate two augmented views
        x1, x2 = x, x  # In practice, apply different augmentations here

        # Forward pass
        z1, z2 = model(x1, x2)

        # Compute contrastive loss
        loss = contrastive_loss(z1, z2, temperature)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
print(f"Training complete after {time.time()-t0}s!")