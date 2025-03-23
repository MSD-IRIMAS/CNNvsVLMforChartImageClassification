# Vision-Language Models for Chart Classification

This directory contains the code and resources for evaluating various Vision-Language Models (VLMs) on the chart image classification task using zero-shot prompting approaches.

## Overview

We evaluate the zero-shot generalization capabilities of several state-of-the-art Vision-Language Models on classifying chart images across 25 different chart types. This allows us to compare the performance of models that have not been specifically trained for chart classification against the CNN models that were trained on our dataset.

## Models Evaluated

We evaluate two categories of Vision-Language Models:

### General-Purpose VLMs

1. **LLaVA Family**:
   - **LLaVA-1.5** (7B and 13B versions): Enhances visual analysis by adopting CLIP-ViT-L-336px and an MLP connector.
   - **LLaVA-1.6** (with Mistral-7B, Vicuna-7B, and Vicuna-13B): Improves visual detail capture through quadrupled resolution and expanded instruction data.

2. **PaLI-GEMMA-3B-ft-VQAv2-448**: Combines a ViT image encoder with a 2B Gemma LLM, fine-tuned on VQAv2.

### Chart-Specific VLMs

1. **ChartLLaMA-13B**: Builds upon LLaVA-1.5's architecture by replacing its single linear projection layer with a two-layer MLP. Specifically trained for chart understanding.

2. **TinyChart-3B-768**: A lightweight approach optimized for chart analysis with a specialized 768×768 resolution and enhanced attention mechanisms for processing structured visual information.

## Directory Structure

```
ChartClassification_with_VLMs/
├── ChartClassification_with_ChartLlama-13b/   # ChartLlama evaluation
├── ChartClassification_with_LLaVA/            # LLaVA family evaluation
├── ChartClassification_with_paligemma-3b-ft-vqav2-448/ # PaLI-GEMMA evaluation
├── ChartClassification_with_TinyChart-3B-768/ # TinyChart evaluation
└── README.md                                  # This documentation file
```

Each subdirectory contains:
- Model-specific code and scripts
- Environment setup instructions (environment.yml)
- Evaluation scripts
- Results in a "review" directory

## Evaluation Approach

Our evaluation follows a consistent approach across all models:

1. **Zero-Shot Prompting**: We test each model using various prompting strategies, from simple questions to detailed descriptions of chart types.
2. **Response Processing**: We process and standardize the model outputs to match our chart type classification scheme.
3. **Metrics Calculation**: We compute standard classification metrics (accuracy, precision, recall, F1-score) and visualize results via confusion matrices.

## Implementation Methods

We use different methods to implement the evaluation for each model family:

1. **ChartLlama & LLaVA**: We clone the official repositories and add a custom inference function.
2. **PaLI-GEMMA**: We use the Hugging Face Transformers library to load and evaluate the model.
3. **TinyChart**: We use the official repository code included in this directory.

## Key Findings

Our experiments show that while Vision-Language Models don't match the performance of specifically trained CNNs on the chart classification task, they demonstrate promising capabilities without task-specific training. Chart-specific VLMs (ChartLlama, TinyChart) generally outperform general-purpose VLMs on this task.

## Getting Started

To evaluate these models, follow the specific setup instructions in each model's subdirectory:

- [ChartLlama-13b Evaluation](./ChartClassification_with_ChartLlama-13b/README.md)
- [LLaVA Family Evaluation](./ChartClassification_with_LLaVA/README.md)
- [PaLI-GEMMA Evaluation](./ChartClassification_with_paligemma-3b-ft-vqav2-448/README.md)
- [TinyChart Evaluation](./ChartClassification_with_TinyChart-3B-768/README.md)

## Implementation Details

All experiments were conducted on an Azure **NC24ads A100 v4** instance equipped with:
- 24-core CPU
- 220 GB of RAM
- NVIDIA A100 graphics card (80 GB memory)

## License Information

The code in this directory is provided for research and educational purposes only under the MIT License. However, each Vision-Language Model has its own license:

- **ChartLlama-13B**: Based on LLaVA and subject to the same license terms as LLaVA.
- **LLaVA models**: Governed by the license specified in the [LLaVA repository](https://github.com/haotian-liu/LLaVA).
- **PaLI-GEMMA**: Made available under Google's model license, which requires acceptance of specific terms.
- **TinyChart**: Released under the license specified in the [TinyChart repository](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart).

When using our evaluation code with any of these models, you must comply with their respective licenses and use the code only for research and educational purposes.

## Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@inproceedings{come2025comparative,
  title={A Comparative Study of CNNs and Vision-Language Models for Chart Image Classification},
  author={Côme, Bruno and Devanne, Maxime and Weber, Jonathan and Forestier, Germain},
  booktitle={Proceedings of the XXth International Conference on Computer Vision Theory and Applications},
  year={2025},
  publisher={SCITEPRESS}
}
```
