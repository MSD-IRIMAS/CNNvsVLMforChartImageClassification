# PaLI-GEMMA Model Evaluation for Chart Classification

This directory contains the code and resources for evaluating the PaLI-GEMMA-3B-ft-VQAv2-448 Vision-Language Model on chart image classification tasks using zero-shot prompting.

## Overview

PaLI-GEMMA is a vision-language model that combines a ViT (Vision Transformer) image encoder with a 2B Gemma language model, fine-tuned on VQAv2. Developed by Google, this model represents a powerful approach to multimodal understanding tasks.

This evaluation framework tests the model's ability to classify chart images into 25 different chart types using various prompting strategies without any task-specific fine-tuning.

## Directory Structure

```
ChartClassification_with_paligemma-3b-ft-vqav2-448/
├── review/                   # Results and evaluation metrics
├── environment.yml           # Conda environment specification
├── metrics_and_conf_matrix.py # Metrics calculation and visualization utilities
├── test_paligemma-3b-ft-vqav2-448.py # Main evaluation script
└── README.md                 # This documentation file
```

## Installation

### Prerequisites

- Anaconda or Miniconda
- Hugging Face account with access to PaLI-GEMMA

### Step 1: Setup Environment

Create and activate the conda environment:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate paligemma-env
```

### Step 2: Hugging Face Access Setup

1. Create or log in to your Hugging Face account at [huggingface.co](https://huggingface.co/)
2. Visit the [PaLI-GEMMA model page](https://huggingface.co/google/paligemma-3b-ft-vqav2-448)
3. Accept Google's usage license for PaLI-GEMMA
4. Generate an access token in your Hugging Face account settings
5. Update the `login(token="YOUR_LOGIN_KEY")` line in the evaluation script with your token

## Usage

### Evaluating PaLI-GEMMA on Chart Classification

To evaluate PaLI-GEMMA on the chart classification task:

```bash
# Make sure you have the test dataset prepared
# The script expects shuffled_input_X_test.npz and shuffled_target_Y_test.npz in the Image_Dataset directory

# Activate the environment
conda activate paligemma-env

# Run the evaluation script
python test_paligemma-3b-ft-vqav2-448.py
```

### Evaluation Process

The evaluation script:

1. Loads the PaLI-GEMMA model from Hugging Face Hub
2. Tests multiple prompting strategies:
   - **First prompt**: Simple query asking for the chart type
   - **Second prompt**: Includes a list of possible chart types
   - **Third prompt**: Detailed instructions on classifying charts
   - **Fourth prompt**: Comprehensive descriptions of each chart type
   - Additional experimental prompts like `context` and `second_2`
3. Processes the model's responses to standardize predictions
4. Calculates performance metrics (accuracy, precision, recall, F1-score)
5. Generates visualizations (confusion matrices, metrics per class)

### Viewing Results

After running the evaluation, the results are stored in the `review` directory:

- `Review_prompt_*.txt`: Detailed evaluation metrics and runtime
- `Confusion_matrix_prompt_*.png`: Confusion matrices for each prompt type
- `metrics_per_class_prompt_*.png`: Bar charts showing metrics for each class
- `Classif_report_prompt_*.csv`: Classification reports in CSV format

## Understanding the Code

### Main Files

- `test_paligemma-3b-ft-vqav2-448.py`: Main script for evaluating PaLI-GEMMA on chart classification
- `metrics_and_conf_matrix.py`: Utility functions for calculating metrics and generating visualizations

### Key Functions

- `run_experiment()`: Runs PaLI-GEMMA with a specific prompt and processes the dataset
- `rectify_preds()`: Standardizes and corrects the model's predictions
- `detect_external_classes()`: Identifies predictions that don't match any known chart type
- `get_all_results()`: Calculates and saves performance metrics

### Customizing Prompts

The script provides several prompt templates that can be modified or extended in the `run_experiment()` function. Each prompt strategy uses a different approach to elicit accurate chart type predictions from the model.

## Model Information

PaLI-GEMMA-3B-ft-VQAv2-448 specifications:
- **Image Encoder**: Vision Transformer (ViT)
- **Language Model**: 2B Gemma LLM
- **Fine-tuning**: The model is fine-tuned on VQAv2 (Visual Question Answering)
- **Resolution**: Supports 448×448 input images
- **Quantization**: Uses BFloat16 for reduced memory usage

## License

This code is provided for research and educational purposes only. PaLI-GEMMA is made available under Google's model license, which you must accept before using the model. Please see the [Hugging Face model page](https://huggingface.co/google/paligemma-3b-ft-vqav2-448) for details.
