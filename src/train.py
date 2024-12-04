import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import os
from model import MNISTNet

# Configuration
class Config:
    BATCH_SIZE = 64
    TEST_BATCH_SIZE = 1000
    EPOCHS = 10
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0

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
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    device = config.DEVICE
    print(f"Using device: {device}")

    # Data loading with augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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

    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, nesterov=True)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE,
                                            epochs=config.EPOCHS, steps_per_epoch=len(train_loader))
    criterion = nn.NLLLoss()

    # Training loop
    best_accuracy = 0
    for epoch in range(1, config.EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, criterion, scheduler)
        accuracy = test(model, device, test_loader, criterion)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save model with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join('models', f'mnist_model_{timestamp}_acc_{accuracy:.2f}.pt')
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main() 