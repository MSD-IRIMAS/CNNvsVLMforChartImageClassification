# Web Scraping Tool for Chart Images

This directory contains the tools used for the first step of our dataset construction process: web scraping chart images from Google Images.

## Overview

The script `seleniumCrawler.py` uses Selenium to automate the process of searching and downloading chart images from Google. It searches for each of the 25 chart types defined in our research and saves the images in their respective class folders.

## Prerequisites

- Python 3.6+
- Selenium
- Chrome WebDriver (must match your Chrome version)
- Pillow (PIL)

## Installation

```bash
# Install required packages
pip install selenium pillow

# Download ChromeDriver from https://sites.google.com/chromium.org/driver/
# Make sure to download the version that matches your Chrome browser
# Place chromedriver.exe in this directory
```

## Usage

To run the web scraping tool:

```bash
python seleniumCrawler.py
```

## Configuration

You can modify the following parameters in the script:

- `download_path`: Directory where images will be saved (default: "chartClasses/")
- `words_to_search`: List of chart types to search for
- `images_nb`: Number of images to download per class (default: 500)
- `first_img_position`: Position of the first image to start downloading (default: 0)

## Process

The script performs the following steps:

1. For each chart type in `words_to_search`:
   - Creates a folder with the chart type name
   - Opens Google Images and searches for the chart type
   - Scrolls down to load more images
   - Downloads up to `images_nb` images for each chart type
   - Saves images in PNG format

2. After all chart types have been processed, converts all PNG images to JPG format

## Output

Images are saved in the following structure:

```
chartClasses/
├── area_chart/
│   ├── area_chart_1.jpg
│   ├── area_chart_2.jpg
│   └── ...
├── bar_chart/
├── ...
└── violin_plot/
```

## Important Notes

- The script includes a delay between actions to avoid being blocked by Google
- Some images may be skipped if they can't be downloaded (e.g., broken links)
- The script provides progress updates during execution
- After web scraping, a manual filtering step is required to remove irrelevant or low-quality images

## Next Steps

After running this script, you should:

1. Manually review all downloaded images
2. Remove irrelevant, misclassified, or low-quality images
3. Move the filtered images to the appropriate folders in the `Image_Dataset/chartClasses/` directory (train, validation, and test)
4. Generate additional images to complete the dataset using the scripts in the `Image_Dataset` directory

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

