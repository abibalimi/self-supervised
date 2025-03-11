import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from transform_helpers import (
    
    augmentation1,
    augmentation2,
)
from pathlib import Path
import matplotlib.pyplot as plt
import time
import numpy as np
from lars_optimizer import LARS



# Hyperparameters
batch_size = 4096
temperature = 0.1
learning_rate = 0.001
base_lr = 0.3 * batch_size / 256  # Learning rate = 4.8
weight_decay = 1e-6
epochs = 100

# Steps:
# 1 - Data Augmentation
# 2 - Encoder Network
# 3 - Projection Head
# 4 - Contrastive Loss





#          ***         Data Augmentation         ***         #
# Custom dataset to apply Albumentations
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augmentation1, augmentation2):
        self.dataset = dataset
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)  # Convert PIL Image to numpy array
        x1 = self.augmentation1(image=image)['image']
        x2 = self.augmentation2(image=image)['image']
        return x1, x2, label



#          ***         Encoder Network (ResNet-18)         ***         #
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = resnet50(weights=None)   # Random initialization
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




def main():
    # Set device
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    # Load CIFAR-10 dataset
    to_load = time.time()
    train_dataset = CIFAR10(root='./data', train=True, download=True)#, transform=transform)
    augmented_dataset = AugmentedDataset(train_dataset, augmentation1, augmentation2)
    train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    print(f"Loading complete after {time.time()-to_load}s!")
    

    # Initialize model, optimizer, and loss
    encoder = Encoder().to(device)
    projection_head = ProjectionHead().to(device)
    model = SimCLR(encoder, projection_head).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Create a directory to save checkpoints
    checkpoint_dir = Path("./simclr_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    # Training loop
    t0_train = time.time()

    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x1, x2, _) in enumerate(train_loader):
            # Move data to device
            x1, x2 = x1.to(device), x2.to(device)
            
            # Forward pass
            z1, z2 = model(x1, x2)

            # Compute contrastive loss
            loss = contrastive_loss(z1, z2, temperature)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        # Log loss
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f"simclr_checkpoint_epoch_{epoch+1}_A.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        
    print(f"Training complete after {time.time()-t0_train}s!")

    # 10 epochs run on mps backend device in ~10 minutes.
    # 20 epochs run on mps backend device in ~27 minutes.


    # Plot learning curve
    plt.plot(range(1, epochs+1), loss_history, label="Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()

    # Save the plot
    plt.savefig("learning_curve_epochs_A.png")  # Save as PNG
    plt.savefig("learning_curve_epochs_A.pdf")  # Save as PDF
    plt.savefig("learning_curve_epochs_A.svg")  # Save as SVG


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support() # required when using the spawn start method.
    main()
