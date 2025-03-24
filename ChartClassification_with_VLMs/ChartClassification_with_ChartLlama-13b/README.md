# ChartLlama-13b Evaluation for Chart Classification

This directory contains the code and resources for evaluating the ChartLlama-13b Vision-Language Model on chart image classification tasks using zero-shot prompting.

## Overview

ChartLlama-13b is a vision-language model specifically trained for chart understanding. It builds upon LLaVA-1.5's architecture by replacing its single linear projection layer with a two-layer MLP and is trained on chart-specific data.

This evaluation framework tests the model's ability to classify chart images into 25 different chart types using various prompting strategies.

## Directory Structure

```
ChartClassification_with_ChartLlama-13b/
├── ChartLlama-13b/          # Directory for ChartLlama-13b model weights and configs
├── llava/                   # LLaVA codebase required for ChartLlama
├── review/                  # Results and evaluation metrics
├── scripts/                 # Scripts from ChartLlama repository
├── environment.yml          # Conda environment specification
├── metrics_and_conf_matrix.py # Metrics calculation and visualization utilities
├── test_ChartLlama-13b.py   # Main evaluation script
└── README.md                # This documentation file
```

## Installation

### Step 1: Get ChartLlama Code

Clone the official ChartLlama repository:

```bash
# Clone the repository (at a location of your choice)
git clone https://github.com/tingxueronghua/ChartLlama-code.git
```

### Step 2: Setup Environment

Follow the environment setup instructions in the official ChartLlama repository README. These instructions will guide you through setting up the proper Python environment with all required dependencies.

Alternatively, you can use our provided environment file:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate chartllama-env
```

### Step 3: Set Up Directory Structure

Copy the required code from the cloned repository:

```bash
# Copy the llava directory
cp -r ChartLlama-code/llava/ ./llava/

# Copy the scripts directory
cp -r ChartLlama-code/scripts/ ./scripts/
```

### Step 4: Download ChartLlama-13b Model

Download the model weights from Hugging Face:

```bash
# Create model directory if it doesn't exist
mkdir -p ChartLlama-13b

# Download model files (following ChartLlama's instructions)
# Place them in the ChartLlama-13b directory
```

The model weights can be found at the Hugging Face repository specified in the original ChartLlama documentation.

### Step 5: Add Custom Inference Function

Add the provided custom inference function to the LLaVA evaluation code:

1. Open the file `llava/eval/model_vqa_lora.py`
2. Add the following function right before the `if __name__ == "__main__":` line:

```python
def custom_inference(image, query, model_path, model, tokenizer,
                     image_processor, temperature=0.2, max_new_tokens=512):
    
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    qs = query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    return outputs
```

## Usage

### Evaluating ChartLlama-13b

To evaluate ChartLlama-13b on the chart classification task:

```bash
# Make sure you have the test dataset prepared
# The script expects shuffled_input_X_test.npz and shuffled_target_Y_test.npz in the Image_Dataset directory

# Activate the environment
conda activate chartllama-env

# Run the evaluation script
python test_ChartLlama-13b.py
```

### Evaluation Process

The evaluation script:

1. Loads the ChartLlama-13b model
2. Tests four different prompting strategies:
   - **First prompt**: Simple query asking for the chart type
   - **Second prompt**: Includes a list of possible chart types
   - **Third prompt**: Detailed instructions on classifying charts
   - **Fourth prompt**: Comprehensive descriptions of each chart type
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

### Main Scripts

- `test_ChartLlama-13b.py`: Main evaluation script that loads the model, runs inference, and processes results
- `metrics_and_conf_matrix.py`: Utility functions for calculating metrics and generating visualizations

### Key Functions

- `run_experiment()`: Runs the model with a specific prompt and processes the dataset
- `rectify_preds()`: Standardizes and corrects the model's predictions
- `detect_external_classes()`: Identifies predictions that don't match any known chart type
- `get_all_results()`: Calculates and saves performance metrics

## License

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). For the full license text, visit: [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/).

This license allows you to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- NonCommercial — You may not use the material for commercial purposes
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original

This code is intended for research and educational purposes only. ChartLlama is based on LLaVA and subject to the same license terms. When using this code, you must comply with the LLaVA license as specified in their repository.
