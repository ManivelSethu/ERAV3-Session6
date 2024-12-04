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