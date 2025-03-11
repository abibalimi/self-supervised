import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from transform_helpers import augmentation1 as augment#, augmentation2 # Leave out the Gaussian blur
from pathlib import Path
import matplotlib.pyplot as plt
import time
import numpy as np



# Hyperparameters
batch_size = 256
base_lr = 0.001#3 * batch_size / 256  # Learning rate = 1.2
weight_decay = 1e-6
temperature = 0.5
epochs = 100

# Steps:
# 1 - Data Augmentation
# 2 - Encoder Network
# 3 - Projection Head
# 4 - Contrastive Loss





#          ***         Data Augmentation         ***         #
# Custom dataset to apply Albumentations
class AugmentedDataset(Dataset):
    def __init__(self, dataset, augment):
        self.dataset = dataset
        self.augmentation = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)  # Convert PIL Image to numpy array
        x1 = self.augmentation(image=image)['image']
        x2 = self.augmentation(image=image)['image']
        return x1, x2, label



#          ***         Encoder Network (ResNet-18)         ***         #
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = resnet18()   # Random initialization
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
def contrastive_loss(z1, z2, temperature=0.1):
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




    
# Load CIFAR-10 dataset
def load_dataset(train=True, batch_size=batch_size):
    """Loads split datasets"""
    dataset = CIFAR10(root='./data', train=train, download=True)
    augmented_dataset = AugmentedDataset(dataset, augment)
    dataset_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataset_loader
    
    
def init_model(device, base_lr, weight_decay):
    """Initializes the model and the optimizer"""
    encoder = Encoder().to(device)
    projection_head = ProjectionHead().to(device)
    model = SimCLR(encoder, projection_head).to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    return model, optimizer
    
    
def lr_scheduler(optimizer, epochs=epochs, warm_up_rate = 0.1):
    """Schedules the learning rate"""
    warmup_epochs = epochs * warm_up_rate # 10% 
    total_epochs = epochs # 100
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    return scheduler, warmup_scheduler, warmup_epochs
    
def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    
    # Load CIFAR-10 dataset
    to_load = time.time()
    
    train_loader = load_dataset(True)
    val_loader = load_dataset(False)
    print(f"Loading complete after {time.time()-to_load}s!")
    

    # Initialize model, optimizer, and loss
    model, optimizer = init_model(device, base_lr, weight_decay)

    
    # Learning rate schedule
    scheduler, warmup_scheduler, warmup_epochs = lr_scheduler(optimizer)


    # Create a directory to save checkpoints
    checkpoint_dir = Path("./simclr_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    #          ***         Training loop         ***         #
    
    # Gradient accumulation parameters
    accumulation_steps = 4  # Simulate batch_size=128 with accumulation
    optimizer.zero_grad()  # Initialize gradients

    train_loss_history = []
    val_loss_history = []
    
    t0_start = time.time()
    for epoch in range(epochs):
        model.train()
        per_epoch_train_loss = 0
        for batch_idx, (x1, x2, _) in enumerate(train_loader):
            # Move data to device and make predictions for this batch (Forward pass)
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1, x2)

            # Compute contrastive loss and its gradients (Backward pass)
            train_loss = contrastive_loss(z1, z2, temperature)
            train_loss = train_loss / accumulation_steps  # Scale loss
            train_loss.backward()
            
            # Update weights after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # Zero the gradients for every batch!
            
            per_epoch_train_loss += train_loss.item()


        # Update learning rate
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Set the model to evaluation/validation mode
        model.eval()
        per_epoch_val_loss = 0
        with torch.no_grad():
            for batch_idx, (x1, x2, _) in enumerate(val_loader):
                # Move data to device and predict
                x1, x2 = x1.to(device), x2.to(device)
                z1, z2 = model(x1, x2)

                # Compute contrastive loss
                val_loss = contrastive_loss(z1, z2, temperature)
                per_epoch_val_loss += val_loss.item()
        
        # Log losses
        avg_train_loss = per_epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        avg_val_loss = per_epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Losses :: Train = {avg_train_loss:.4f} , Val = {avg_val_loss:.4f}")      
                
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f"simclr_checkpoint_epoch_{epoch+1}_A_val.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")       
    
         # Free up memory
        torch.mps.empty_cache()
    
    
    print(f"Training + validation completed after {time.time()-t0_start}s!")
    # 10 epochs run on mps backend device in ~10 minutes.(37 minutes with val)
    # 20 epochs run on mps backend device in ~27 minutes.
    
    


    # Plot learning curve
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, train_loss_history, label="Train")
    plt.plot(epoch_range, val_loss_history, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()

    # Save the plot
    plt.savefig("learning_curve_epochs_reproducing_SimCLR_CIFAR-10_ResNet18.png")  # Save as PNG


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support() # required when using the spawn start method.
    main()