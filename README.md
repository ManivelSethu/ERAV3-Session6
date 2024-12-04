# MNIST CNN with CI/CD Pipeline

A lightweight Convolutional Neural Network for MNIST digit classification with automated testing and CI/CD pipeline.

## Project Structure
```
.
├── src/
│   ├── model.py      # CNN model architecture
│   └── train.py      # Training script
├── tests/
│   └── test_model.py # Model tests
├── data/             # MNIST dataset (auto-downloaded)
├── models/           # Saved model checkpoints
└── .github/
    └── workflows/    # GitHub Actions workflow
```

## Test Logs
Total parameters: 15,334
Has Batch Normalization: True
Has Dropout: True
Has Fully Connected Layer: True

Epoch 1: 100%|████████████████████████████████████████████████| 469/469 [02:31<00:00,  3.10it/s, Loss=0.5379, Acc=82.73%] 
Test set: Average loss: 0.0546, Accuracy: 9842/10000 (98.42%)
Model saved to models\mnist_model_20241204_164239_acc_98.42.pt

Epoch 2: 100%|████████████████████████████████████████████████| 469/469 [02:30<00:00,  3.12it/s, Loss=0.1408, Acc=95.72%] 
Test set: Average loss: 0.0365, Accuracy: 9883/10000 (98.83%)
Model saved to models\mnist_model_20241204_164516_acc_98.83.pt

Epoch 3: 100%|████████████████████████████████████████████████| 469/469 [02:32<00:00,  3.08it/s, Loss=0.1082, Acc=96.83%]
Test set: Average loss: 0.0378, Accuracy: 9875/10000 (98.75%)

Epoch 4: 100%|████████████████████████████████████████████████| 469/469 [02:29<00:00,  3.13it/s, Loss=0.0866, Acc=97.37%] 
Test set: Average loss: 0.0260, Accuracy: 9914/10000 (99.14%)
Model saved to models\mnist_model_20241204_165031_acc_99.14.pt

Epoch 5: 100%|████████████████████████████████████████████████| 469/469 [02:30<00:00,  3.12it/s, Loss=0.0737, Acc=97.81%] 
Test set: Average loss: 0.0211, Accuracy: 9933/10000 (99.33%)
Model saved to models\mnist_model_20241204_165308_acc_99.33.pt

Epoch 6: 100%|████████████████████████████████████████████████| 469/469 [02:29<00:00,  3.13it/s, Loss=0.1014, Acc=96.95%] 
Test set: Average loss: 0.0287, Accuracy: 9910/10000 (99.10%)

Epoch 7: 100%|████████████████████████████████████████████████| 469/469 [02:29<00:00,  3.13it/s, Loss=0.0911, Acc=97.31%] 
Test set: Average loss: 0.0272, Accuracy: 9914/10000 (99.14%)

Epoch 8: 100%|████████████████████████████████████████████████| 469/469 [02:30<00:00,  3.13it/s, Loss=0.0760, Acc=97.71%] 

Test set: Average loss: 0.0269, Accuracy: 9908/10000 (99.08%)

Epoch 9: 100%|████████████████████████████████████████████████| 469/469 [02:29<00:00,  3.15it/s, Loss=0.0676, Acc=98.00%] 
Test set: Average loss: 0.0190, Accuracy: 9936/10000 (99.36%)

Model saved to models\mnist_model_20241204_170332_acc_99.36.pt
Epoch 10: 100%|███████████████████████████████████████████████| 469/469 [02:29<00:00,  3.13it/s, Loss=0.0574, Acc=98.23%] 

Test set: Average loss: 0.0161, **Accuracy: 9948/10000 (99.48%)**
Model saved to models\mnist_model_20241204_170609_acc_99.48.pt

Reached target accuracy of 99.4%! Training completed early at **epoch 10**

## Features
- Efficient CNN architecture (<20K parameters)
- Batch Normalization and Dropout for regularization
- Automated testing pipeline
- Model validation checks
- Configurable hyperparameters
- Training progress with tqdm
- Timestamped model checkpoints

## Requirements
- Python 3.9+
- PyTorch
- torchvision
- tqdm
- pytest

## Local Setup and Training

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
python -m pytest tests/test_model.py -v
```

4. Train the model:
```bash
python src/train.py
```

## Model Architecture
- Input: 28x28 grayscale images
- 3 Convolutional layers with Batch Normalization
- Max Pooling layers
- Dropout for regularization
- Fully Connected layers
- Output: 10 classes (digits 0-9)

## CI/CD Pipeline
The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs dependencies
3. Runs model tests
4. Validates model architecture
5. Checks parameter count
6. Verifies input/output dimensions

## Notes
- Model checkpoints are saved with timestamps and accuracy metrics
- Dataset and models are excluded from version control (see .gitignore)
- Training can be done on CPU or GPU (automatically detected) 
