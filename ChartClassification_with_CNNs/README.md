# Chart Classification with CNNs

This directory contains the scripts for training and evaluating Convolutional Neural Networks (CNNs) on the chart image classification task.

## Overview

In this study, we train and evaluate six prominent CNN architectures that have demonstrated significant success in various image classification tasks:

1. **AlexNet** - The pioneering deep CNN architecture, consisting of five convolutional layers followed by three fully connected layers.
2. **VGG16** - A deeper architecture with 16 layers using small 3×3 convolution filters throughout the network.
3. **Inception-v3** - Employs parallel convolution paths of varying scales within its Inception modules for multi-scale feature processing.
4. **Inception-ResNet-v2** - Combines the Inception modules with residual connections, enhancing gradient flow and feature extraction.
5. **Xception** - Leverages depthwise separable convolutions to efficiently process cross-channel and spatial correlations.
6. **EfficientNetB4** - A scaled version of EfficientNet optimized through neural architecture search, offering state-of-the-art performance with fewer parameters.

## Directory Structure

```
ChartClassification_with_CNNs/
├── dtnls_models/               # Custom model implementations
│   ├── __init__.py
│   └── dtnls_alexnet.py        # AlexNet implementation
├── saved_models/               # Directory for storing trained models and results
├── cnns_training.py            # Script containing functions for model training and evaluation
├── environment.yml             # Conda environment specification
├── main.py                     # Main script to execute training and evaluation
├── plot_confusion_matrix.py    # Utility for plotting confusion matrices and metrics
└── README.md                   # This documentation file
```

## Installation

This project uses a conda environment for managing dependencies. To set up the environment:

```bash
# Create and activate the conda environment from the environment.yml file
conda env create -f environment.yml
conda activate tensorEnv
```

### Prerequisites

- Input data files (generated from the Image_Dataset directory)

## Usage

### Training and Evaluating Models

The main script (`main.py`) handles the loading of data, setting up models, and executing training and evaluation. To run the script:

```bash
python main.py
```

By default, this will train all six CNN models with full training. You can modify the `main.py` file to change:

- Which models to train (`all_models` list)
- Training type ('full' or 'fine-tuning')
- Batch size and number of epochs
- Whether to train all models or a single model

### Configuration Options

Edit `main.py` to configure the following options:

```python
# Training type: 'full' or 'fine-tuning'
training_type = 'full'

# List of models to train
all_models = ['dtnls_alexnet', 'VGG16', 'InceptionV3', 'Xception', 'InceptionResNetV2', 'EfficientNetB4']

# Batch size and number of epochs
batch_size = 64
epochs = 100

# Choose between training all models or a single model
cnn_train.train_all_models(...)  # Train all models
# cnn_train.train_single_model(...)  # Train a single model
```

### Training Modes

The script supports two training modes:

1. **Full Training** (`training_type = 'full'`): Trains the models from scratch with random initialization.
2. **Fine-Tuning** (`training_type = 'fine-tuning'`): Leverages pre-trained weights from ImageNet and fine-tunes the models for chart classification.

### Model Outputs

After training, the following outputs are generated in the `saved_models/` directory:

- Trained model files (`.keras`)
- Training and validation loss curves (`.png`)
- Training and validation accuracy curves (`.png`)
- Confusion matrices (`.png`)
- Detailed performance reports (`.txt`)
- Classification metrics per class (`.png`)
- Classification report in CSV format (`.csv`)

## Scripts

### main.py

The main script that orchestrates the entire training and evaluation process:
- Sets up GPU configuration
- Loads the preprocessed data
- Defines the chart classes
- Configures and executes model training

### cnns_training.py

Contains the core functions for model training and evaluation:
- `preprocess_data_for_model`: Preprocesses images according to each model's requirements
- `set_model`: Creates model architectures (from scratch or with pretrained weights)
- `train_full_model`: Trains models from scratch
- `fine_tune_model`: Fine-tunes pretrained models
- `model_evaluation_and_review`: Evaluates models and generates performance reports
- `train_single_model`: Trains a single specified model
- `train_all_models`: Trains all models in the provided list

### plot_confusion_matrix.py

Utility script for generating confusion matrices and performance metrics:
- `matrice_confusion`: Generates confusion matrix visualizations
- `get_all_results`: Calculates and saves various performance metrics
- `Y_array_to_class_label_array`: Converts numerical labels to class names

### dtnls_models/dtnls_alexnet.py

Custom implementation of the AlexNet architecture, adapted for the chart classification task.

## Expected Results

After training, the models should achieve different accuracy levels on the chart classification task. Overall, the more recent architectures (like EfficientNetB4) tend to outperform older ones (like AlexNet) due to their more sophisticated architectures.

## License

This code is provided for research and educational purposes only under the MIT License.
