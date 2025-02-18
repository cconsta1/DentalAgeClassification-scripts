# Vision Transformer (ViT) & EfficientNetV2 Models for Age Classification

## Overview

This repository contains implementations of Vision Transformer (ViT) and EfficientNetV2 models for binary age classification. The models are trained to classify whether an individual is below 18 years old (0) or 18 and above (1) using panoramic dental x-rays.

These scripts use PyTorch and Torchvision to leverage pre-trained models with custom classification heads, and they train the models with binary cross-entropy loss (BCEWithLogitsLoss) over 25 epochs.

## Features

- Uses ViT-B-16 and EfficientNetV2-S from `torchvision.models`
- Supports training with and without data augmentation
- Applies data transformations and normalization for training stability
- Supports GPU acceleration for faster training
- Implements data filtering to train only on ages 14-24
- Uses AdamW optimizer with weight decay for stability
- Stores model checkpoints during training

## Installation

Ensure you have the necessary dependencies installed:

```bash
pip install torch torchvision numpy pandas scikit-learn pillow matplotlib
```

## How to Run the Scripts

1. **Prepare your dataset**: Place your dataset under the `./data/DentAgePooledDatav2/` directory or set the `DATA_PATH` environment variable.
2. **Run the desired training script**:

   - **Vision Transformer (ViT) - No Augmentation**
     ```bash
     python model_vit_no_augmentation.py
     ```

   - **Vision Transformer (ViT) - With Augmentation**
     ```bash
     python model_vit_augmented.py
     ```

   - **EfficientNetV2 - No Augmentation**
     ```bash
     python model_efficientnetv2_no_augmentation.py
     ```

   - **EfficientNetV2 - With Augmentation**
     ```bash
     python model_efficientnetv2_augmented.py
     ```

3. **Monitor training output**: Loss and progress will be displayed on the console.
4. **Check model checkpoints**: Trained models are saved in the corresponding `./models/` directory.

## Model Architectures

### Vision Transformer (ViT)
- A pre-trained ViT-B-16 backbone
- A custom classifier head with multiple fully connected layers and dropout
- An output layer with 1 neuron for binary classification

### EfficientNetV2
- A pre-trained EfficientNetV2-S backbone
- A custom classification head with multiple fully connected layers and dropout
- An output layer with 1 neuron for binary classification

## Configuration

The scripts use environment variables for flexibility:

- `DATA_PATH`: Path to the dataset (default: `./data`)
- `MODEL_PATH`: Path to save models and logs (default: `./models/`)

## Training Details

- **Batch size**: 16
- **Epochs**: 25
- **Optimizer**: AdamW (`lr=5e-6`, `weight_decay=5e-7`)
- **Loss function**: BCEWithLogitsLoss
- **GPU support**: Yes (CUDA enabled)

## License

This project is open-source under the MIT License.

## Contact

For questions, reach out via GitHub issues or discussions.
