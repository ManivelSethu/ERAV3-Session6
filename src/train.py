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

class Logger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        
    def log(self, message, print_to_console=True):
        if print_to_console:
            print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

# Configuration
class Config:
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 1000
    EPOCHS = 10  #  epochs
    LEARNING_RATE = 0.002  #  learning rate
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

def print_model_summary(model, logger):
    summary = [
        "\n" + "="*75,
        "Model Summary",
        "="*75,
        "{:<30} {:<20} {:>10}".format("Layer (type)", "Output Shape", "Param #"),
        "-"*75
    ]
    
    total_params = 0
    trainable_params = 0
    
    def get_layer_info(layer, input_size):
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            params = sum(p.numel() for p in layer.parameters())
            trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            return params, trainable
        return 0, 0
    
    # Process Sequential blocks
    def process_sequential(module, input_size):
        nonlocal total_params, trainable_params
        current_size = input_size
        
        for layer in module:
            if isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                kernel_size = layer.kernel_size[0]
                padding = layer.padding[0]
                stride = layer.stride[0]
                h = (current_size[2] + 2*padding - kernel_size) // stride + 1
                w = (current_size[3] + 2*padding - kernel_size) // stride + 1
                current_size = (current_size[0], out_channels, h, w)
                params, trainable = get_layer_info(layer, current_size)
                print("{:<30} {:<20} {:>10,d}".format(
                    f"Conv2d-{out_channels}", str(list(current_size)), params))
                total_params += params
                trainable_params += trainable
                
            elif isinstance(layer, nn.BatchNorm2d):
                params, trainable = get_layer_info(layer, current_size)
                print("{:<30} {:<20} {:>10,d}".format(
                    f"BatchNorm2d-{current_size[1]}", str(list(current_size)), params))
                total_params += params
                trainable_params += trainable
                
            elif isinstance(layer, nn.MaxPool2d):
                h = current_size[2] // layer.kernel_size
                w = current_size[3] // layer.kernel_size
                current_size = (current_size[0], current_size[1], h, w)
                print("{:<30} {:<20} {:>10,d}".format(
                    "MaxPool2d", str(list(current_size)), 0))
                
            elif isinstance(layer, nn.Dropout):
                print("{:<30} {:<20} {:>10,d}".format(
                    f"Dropout-{layer.p}", str(list(current_size)), 0))
                
            elif isinstance(layer, nn.ReLU):
                print("{:<30} {:<20} {:>10,d}".format(
                    "ReLU", str(list(current_size)), 0))
                
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                current_size = (current_size[0], current_size[1], 1, 1)
                print("{:<30} {:<20} {:>10,d}".format(
                    "AdaptiveAvgPool2d", str(list(current_size)), 0))
    
    # Start with input size
    x = torch.randn(1, 1, 28, 28)
    current_size = x.size()
    
    # Process main layers
    process_sequential(model.conv1, current_size)
    current_size = (current_size[0], 10, 12, 12)  # After conv1
    process_sequential(model.conv2, current_size)
    current_size = (current_size[0], 16, 4, 4)    # After conv2
    process_sequential(model.conv3, current_size)
    current_size = (current_size[0], 10, 1, 1)    # After conv3
    
    # Process final linear layer
    params, trainable = get_layer_info(model.fc, (1, 10))
    print("{:<30} {:<20} {:>10,d}".format(
        "Linear", "(1, 10)", params))
    total_params += params
    trainable_params += trainable
    
    print("="*75)
    print("{:<50} {:>10,d}".format("Total params", total_params))
    print("{:<50} {:>10,d}".format("Trainable params", trainable_params))
    print("{:<50} {:>10,d}".format("Non-trainable params", total_params - trainable_params))
    print("="*75 + "\n")
    
    for line in summary:
        logger.log(line)

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
    
    return running_loss/(batch_idx+1), 100.*correct/total

def test(model, device, test_loader, criterion, logger):
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
    
    # Log validation results
    logger.log("\n" + "="*50)
    logger.log("Test/Validation Results:")
    logger.log(f"Average loss: {test_loss:.4f}")
    logger.log(f"Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    logger.log("="*50 + "\n")
    
    return accuracy, test_loss

def main():
    config = Config()
    logger = Logger()
    
    # Log training configuration
    logger.log("\n" + "="*50)
    logger.log("Training Configuration:")
    logger.log(f"Random Seed: {config.SEED}")
    logger.log(f"Batch Size: {config.BATCH_SIZE}")
    logger.log(f"Learning Rate: {config.LEARNING_RATE}")
    logger.log(f"Epochs: {config.EPOCHS}")
    logger.log("="*50 + "\n")
    
    print(f"Using random seed: {config.SEED}")
    set_seed(config.SEED)
    
    device = config.DEVICE
    logger.log(f"Using device: {device}")

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
    
    # Print detailed model summary
    print_model_summary(model, logger)
    
    # Log basic model information
    logger.log(f"Has Batch Normalization: {model.has_batch_norm()}")
    logger.log(f"Has Dropout: {model.has_dropout()}")
    logger.log(f"Has Fully Connected Layer: {model.has_fc_layer()}\n")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-6
    )
    criterion = nn.NLLLoss()

    # Training loop with patience for early stopping
    best_accuracy = 0
    patience = 5
    patience_counter = 0
    
    logger.log("\n" + "="*50)
    logger.log("Starting Training")
    logger.log("="*50)
    
    # Create a summary table for epoch results
    logger.log("\nEpoch Results Summary:")
    logger.log("{:<6} {:<12} {:<12} {:<12} {:<12}".format(
        "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"))
    logger.log("-"*54)
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, criterion, scheduler)
        val_accuracy, val_loss = test(model, device, test_loader, criterion, logger)
        
        # Log epoch summary
        logger.log("{:<6d} {:<12.4f} {:<12.2f} {:<12.4f} {:<12.2f}".format(
            epoch, train_loss, train_acc, val_loss, val_accuracy))
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join('models', f'mnist_model_{timestamp}_acc_{val_accuracy:.2f}.pt')
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.log(f"\nNew best model saved!")
            logger.log(f"Path: {model_path}")
            logger.log(f"Best accuracy so far: {best_accuracy:.2f}%\n")
            
            if val_accuracy >= 99.4:
                logger.log("\n" + "="*50)
                logger.log(f"Target accuracy of 99.4% reached!")
                logger.log(f"Final test accuracy: {val_accuracy:.2f}%")
                logger.log(f"Training completed at epoch {epoch}")
                logger.log("="*50 + "\n")
                break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.log("\n" + "="*50)
                logger.log(f"No improvement for {patience} epochs.")
                logger.log(f"Early stopping at epoch {epoch}")
                logger.log(f"Best accuracy achieved: {best_accuracy:.2f}%")
                logger.log("="*50 + "\n")
                break
    
    logger.log("\n" + "="*50)
    logger.log("Training Summary:")
    logger.log(f"Best accuracy achieved: {best_accuracy:.2f}%")
    logger.log(f"Random seed used: {config.SEED}")
    logger.log("="*50 + "\n")

if __name__ == '__main__':
    main() 