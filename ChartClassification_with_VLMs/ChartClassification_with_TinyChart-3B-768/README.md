# TinyChart-3B-768 Evaluation for Chart Classification

This directory contains the code and resources for evaluating the TinyChart-3B-768 Vision-Language Model on chart image classification tasks using zero-shot prompting.

## Overview

TinyChart-3B-768 is a specialized vision-language model optimized for chart analysis with a 768×768 pixel resolution and enhanced attention mechanisms for processing structured visual information. It represents a lightweight approach designed specifically for understanding and analyzing charts and visualizations.

This evaluation framework tests the model's ability to classify chart images into 25 different chart types without any task-specific fine-tuning.

## Directory Structure

```
ChartClassification_with_TinyChart-3B-768/
├── review/                   # Results and evaluation metrics
├── scripts/                  # Scripts from TinyChart repository
├── tinychart/                # TinyChart model code
├── app.py                    # Demo application from TinyChart repository
├── environment.yml           # Conda environment specification
├── metrics_and_conf_matrix.py # Metrics calculation and visualization utilities
├── test_TinyChart-3B-768.py  # Main evaluation script
└── README.md                 # This documentation file
```

## Installation

### Step 1: Get TinyChart Code

The necessary TinyChart code has already been included in this directory. The `tinychart` and `scripts` directories, as well as the `app.py` file, were copied from the official TinyChart repository.

For reference, the original TinyChart repository is available at: [X-PLUG/mPLUG-DocOwl/TinyChart](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart)

### Step 2: Setup Environment

Follow the environment setup instructions in the official TinyChart repository README. These instructions will guide you through setting up the proper Python environment with all required dependencies.

Alternatively, you can use our provided environment file:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate tinychart-env
```

### Step 3: Download Model Weights

Follow the instructions in the original TinyChart repository to download the model weights and place them in the `TinyChart-3B-768` directory.

### Step 4: Add Custom Inference Function

Add a custom inference function to the TinyChart evaluation code:

1. Open the file `tinychart/eval/run_tiny_chart.py`
2. Add the following function before the `if __name__ == "__main__":` line:

```python
def custom_inference(image, query, model, tokenizer, image_processor, context_len, conv_mode, temperature=0.2, max_new_tokens=100):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(
        image,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16 if "cuda" in str(model.device) else torch.float32)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            # top_p=top_p,
            # num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    
    return outputs
```

## Usage

### Evaluating TinyChart-3B-768

To evaluate TinyChart-3B-768 on the chart classification task:

```bash
# Make sure you have the test dataset prepared
# The script expects shuffled_input_X_test.npz and shuffled_target_Y_test.npz in the Image_Dataset directory

# Activate the environment
conda activate tinychart-env

# Run the evaluation script
python test_TinyChart-3B-768.py
```

### Evaluation Process

The evaluation script:

1. Loads the TinyChart-3B-768 model
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

### Main Files

- `test_TinyChart-3B-768.py`: Main script for evaluating TinyChart on chart classification
- `metrics_and_conf_matrix.py`: Utility functions for calculating metrics and generating visualizations

### Key Functions

- `run_experiment()`: Runs the model with a specific prompt and processes the dataset
- `rectify_preds()`: Standardizes and corrects the model's predictions
- `detect_external_classes()`: Identifies predictions that don't match any known chart type
- `get_all_results()`: Calculates and saves performance metrics

### TinyChart Model Architecture

TinyChart-3B-768 features:
- A specialized 768×768 pixel resolution for detailed chart understanding
- Enhanced attention mechanisms for processing structured visual information
- A lightweight 3B parameter architecture
- Optimized performance for chart analysis tasks

## License

This code is provided for research and educational purposes only. TinyChart is released under the license specified in the original repository. When using this code with TinyChart, you must comply with their licensing terms.
