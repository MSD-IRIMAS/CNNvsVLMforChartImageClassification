# Chart Image Dataset

This directory contains the complete dataset of chart images and the tools to prepare them for supervised learning.

## Dataset Structure

The dataset consists of 25 chart classes, each with 1,000 images, organized as follows:

```
chartClasses/
├── test/
│   ├── area/
│   ├── bar/
│   ├── ...
├── train/
│   ├── area/
│   ├── bar/
│   ├── ...
└── validation/
│   ├── area/
│   ├── bar/
│   └── ...
```

The chart types included are:
- area
- bar
- barcode_plot
- boxplot
- bubble
- column
- diverging_bar
- diverging_stacked_bar
- donut
- dot_strip_plot
- heatmap
- line
- line_column
- lollipop
- ordered_bar
- ordered_column
- paired_bar
- paired_column
- pie
- population_pyramid
- proportional_stacked_bar
- scatter_plot
- spine
- stacked_column
- violin_plot

## Dataset Composition

The dataset combines:
1. Web-scraped images from Google (collected using the script in the `selenium_crawler` directory)
2. Manually filtered images to ensure quality and relevance
3. Automatically generated images using Python (Matplotlib) and Julia (Plots, Vegalite, Gadfly)

The exact composition of each chart class varies, as detailed in our paper and visualized in the `assets/dataset_composition.png` file in the root directory.

## Data Preparation

The script `create_save_supervised_data.py` processes the images into a format suitable for training and evaluating machine learning models.

### Usage

```bash
python create_save_supervised_data.py
```

### Process

The script:
1. Loads images from the `chartClasses` directories (test, train, and validation)
2. Resizes all images to 400x400 pixels
3. Converts images to arrays
4. Shuffles the data
5. Saves the processed data as compressed NumPy arrays

### Output Files

The script generates the following files:
- `shuffled_input_X_train.npz`: Training images
- `shuffled_target_Y_train.npz`: Training labels
- `shuffled_input_X_validation.npz`: Validation images
- `shuffled_target_Y_validation.npz`: Validation labels
- `shuffled_input_X_test.npz`: Test images
- `shuffled_target_Y_test.npz`: Test labels

These files are used by the CNN training scripts in the `ChartClassification_with_CNNs` directory and by the VLM evaluation scripts in the `ChartClassification_with_VLMs` directory.

## Dataset Creation Workflow

To use and expand this dataset:

1. **Web Scraping** (first step):
   - Use the `seleniumCrawler.py` script from the `selenium_crawler` directory to collect images from Google

2. **Manual Filtering** (second step):
   - Review the scraped images
   - Remove irrelevant, misclassified, or low-quality images
   - Move the filtered images to the appropriate folders in this directory

3. **Data Organization**:
   - Organize filtered images into the appropriate class folders within the test, train, and validation directories
   - Ensure a balanced distribution across classes

4. **Data Preparation**:
   - Run the `create_save_supervised_data.py` script to prepare the data for model training

## Memory Management

Note that the script uses memory optimization techniques (garbage collection and sequential processing) due to the large size of the dataset. Depending on your system's RAM, you may need to adjust the sleep times or process smaller batches.

## License

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). For the full license text, visit: [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/).

This license allows you to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- NonCommercial — You may not use the material for commercial purposes
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original

This code is intended for research and educational purposes only.
