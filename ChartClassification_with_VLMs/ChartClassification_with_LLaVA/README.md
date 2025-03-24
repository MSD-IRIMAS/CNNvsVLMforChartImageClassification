# LLaVA Models Evaluation for Chart Classification

This directory contains the code and resources for evaluating multiple variants of LLaVA Vision-Language Models on chart image classification tasks using zero-shot prompting.

## Overview

LLaVA (Large Language and Vision Assistant) is a leading vision-language model that combines visual understanding with language capabilities. We evaluate five variants from the LLaVA family:

1. LLaVA-1.5-7B
2. LLaVA-1.5-13B
3. LLaVA-1.6-mistral-7B
4. LLaVA-1.6-vicuna-7B
5. LLaVA-1.6-vicuna-13B

These tests explore the zero-shot chart classification capabilities of different model sizes and architectures within the LLaVA family.

## Directory Structure

```
ChartClassification_with_LLaVA/
├── LLaVA/                    # Official LLaVA repository code
├── review/                   # Results and evaluation metrics
│   ├── llava-v1.5-7b/        # Results for LLaVA-1.5-7B
│   ├── llava-v1.5-13b/       # Results for LLaVA-1.5-13B
│   ├── llava-v1.6-mistral-7b/# Results for LLaVA-1.6-mistral-7B
│   ├── llava-v1.6-vicuna-7b/ # Results for LLaVA-1.6-vicuna-7B
│   └── llava-v1.6-vicuna-13b/# Results for LLaVA-1.6-vicuna-13B
├── environment.yml           # Conda environment specification
├── metrics_and_conf_matrix.py # Metrics calculation and visualization utilities
├── test_llava-v1.5-7b.py     # Evaluation script for LLaVA-1.5-7B
├── test_llava-v1.5-13b.py    # Evaluation script for LLaVA-1.5-13B
├── test_llava-v1.6-mistral-7b.py # Evaluation script for LLaVA-1.6-mistral-7B
├── test_llava-v1.6-vicuna-7b.py # Evaluation script for LLaVA-1.6-vicuna-7B
├── test_llava-v1.6-vicuna-13b.py # Evaluation script for LLaVA-1.6-vicuna-13B
└── README.md                 # This documentation file
```

## Installation

### Step 1: Get LLaVA Code

Clone the official LLaVA repository:

```bash
# Clone the repository directly to the LLaVA directory
git clone https://github.com/haotian-liu/LLaVA.git
```

### Step 2: Setup Environment

Follow the environment setup instructions in the official LLaVA repository README. These instructions will guide you through setting up the proper Python environment with all required dependencies.

Alternatively, you can use our provided environment file:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate llava-env
```

### Step 3: Add Custom Inference Function

Add a custom inference function to the LLaVA evaluation code:

1. Open or create the file `LLaVA/llava/eval/run_llava.py`
2. Add the following function before the `if __name__ == "__main__":` line:

```python
def custom_inference(images, query, model_path, model, tokenizer,
                     image_processor, temperature=0.2, max_new_tokens=512):
    
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

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
    
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
        tokenizer.padding_side = "left"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
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
    
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    attention_mask = None
    if "mistral" in model_name.lower():
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass the attention mask
            images=images_tensor,
            image_sizes=image_sizes,
            pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            # top_p=top_p,
            # num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs
```

## Usage

### Evaluating LLaVA Models

To evaluate any of the LLaVA models on the chart classification task:

```bash
# Make sure you have the test dataset prepared
# The script expects shuffled_input_X_test.npz and shuffled_target_Y_test.npz in the Image_Dataset directory

# Activate the environment
conda activate llava-env

# Run the evaluation script for a specific model version
python test_llava-v1.5-7b.py    # For LLaVA-1.5-7B
python test_llava-v1.5-13b.py   # For LLaVA-1.5-13B
python test_llava-v1.6-mistral-7b.py  # For LLaVA-1.6-mistral-7B
python test_llava-v1.6-vicuna-7b.py   # For LLaVA-1.6-vicuna-7B
python test_llava-v1.6-vicuna-13b.py  # For LLaVA-1.6-vicuna-13B
```

### Models and HuggingFace Paths

Each script is configured to use a specific LLaVA variant from HuggingFace:

| Model | HuggingFace Path |
|-------|------------------|
| LLaVA-1.5-7B | liuhaotian/llava-v1.5-7b |
| LLaVA-1.5-13B | liuhaotian/llava-v1.5-13b |
| LLaVA-1.6-mistral-7B | liuhaotian/llava-v1.6-mistral-7b |
| LLaVA-1.6-vicuna-7B | liuhaotian/llava-v1.6-vicuna-7b |
| LLaVA-1.6-vicuna-13B | liuhaotian/llava-v1.6-vicuna-13b |

### Evaluation Process

Each evaluation script:

1. Loads the specified LLaVA model
2. Tests four different prompting strategies (with a slightly different approach for mistral-based models):
   - **First prompt**: Simple query asking for the chart type
   - **Second prompt**: Includes a list of possible chart types
   - **Third prompt**: Detailed instructions on classifying charts
   - **Fourth prompt**: Comprehensive descriptions of each chart type
3. Processes the model's responses to standardize predictions
4. Calculates performance metrics (accuracy, precision, recall, F1-score)
5. Generates visualizations (confusion matrices, metrics per class)

### Viewing Results

After running the evaluation, the results are stored in the `review` directory, with a dedicated subdirectory for each model. Each directory contains:

- `Review_prompt_*.txt`: Detailed evaluation metrics and runtime
- `Confusion_matrix_prompt_*.png`: Confusion matrices for each prompt type
- `metrics_per_class_prompt_*.png`: Bar charts showing metrics for each class
- `Classif_report_prompt_*.csv`: Classification reports in CSV format

## Understanding the Code

### Main Scripts

- `test_llava-v*.py`: Model-specific evaluation scripts that load the LLaVA variants, run inference, and process results
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

This code is intended for research and educational purposes only. The LLaVA models are governed by the license specified in their original repository. When using this code with LLaVA models, you must comply with their licensing terms.
