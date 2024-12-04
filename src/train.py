import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import os
import random
import numpy as np
from model import MNISTNet

# Configuration
class Config:
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 1000
    EPOCHS = 25  # Increased epochs
    LEARNING_RATE = 0.002  # Slightly increased learning rate
    SEED = random.randint(1, 10000)  # Random seed
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, device, train_loader, optimizer, epoch, criterion, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    if scheduler is not None:
        scheduler.step()

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    config = Config()
    print(f"Using random seed: {config.SEED}")
    set_seed(config.SEED)
    
    device = config.DEVICE
    print(f"Using device: {device}")

    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('data', train=False, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type=='cuda'
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type=='cuda'
    )

    # Model setup
    model = MNISTNet().to(device)
    
    # Print model information
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {model.get_num_parameters():,}")
    print(f"Has Batch Normalization: {model.has_batch_norm()}")
    print(f"Has Dropout: {model.has_dropout()}")
    print(f"Has Fully Connected Layer: {model.has_fc_layer()}\n")

    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=1,
        eta_min=1e-6
    )
    
    criterion = nn.NLLLoss()

    # Training loop with patience for early stopping
    best_accuracy = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, criterion, scheduler)
        accuracy = test(model, device, test_loader, criterion)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            # Save model with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join('models', f'mnist_model_{timestamp}_acc_{accuracy:.2f}.pt')
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            if accuracy >= 99.4:
                print(f"\nReached target accuracy of 99.4%! Training completed early at epoch {epoch}")
                break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nNo improvement for {patience} epochs. Early stopping at epoch {epoch}")
                break

if __name__ == '__main__':
    main() 