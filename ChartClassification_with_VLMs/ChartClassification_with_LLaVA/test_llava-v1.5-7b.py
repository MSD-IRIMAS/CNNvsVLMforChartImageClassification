import sys
import os
import time
import datetime as dt
from PIL import Image
import numpy as np
import pandas as pd
import metrics_and_conf_matrix as mcm

sys.path.append(os.path.join(os.path.dirname(__file__), "LLaVA/llava"))

from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.run_llava import custom_inference


def convert_in_lowercase(pred_class):
    lowercase_pred = ""
    for c in pred_class:
        if c.isupper():
            lowercase_pred += c.lower()
        else:
            lowercase_pred += c
    return lowercase_pred


def detect_external_classes(llm_preds, classes_name, reviewFilePath):
    set_of_classes_name = set(classes_name)
    set_of_llm_preds = set(llm_preds)
    # Finds elements that don't belong to our set of chart classes
    extern_cl = set_of_llm_preds - set_of_classes_name
    l_extern_cl = list(extern_cl)
    
    if reviewFilePath is not None:
        try:
            with open(reviewFilePath, 'a') as review:
                review.write('--------- External classes --------\n')
                review.write('In "Other":\n')
                review.write('\n')
                review.write(str(l_extern_cl) + "\n")
                review.write('\n')
                review.close()
        finally:
            review.close()
    else:
        print(l_extern_cl)
    
    return l_extern_cl


def replace_external_classes_by_other(Y_pred, Y_true, external_classes, reviewFilePath):
    set_ext_classes = set(external_classes)
    result = ["other" if elt in set_ext_classes else elt for elt in Y_pred]
    predicted_external_classes = [pred for pred in Y_pred if pred in set_ext_classes]
    corresponding_truth = [true_class for pred, true_class in zip(result, Y_true) if pred == "other"]
    d = {'Predicted external classes': predicted_external_classes, 'Real corresponding classes': corresponding_truth}
    df = pd.DataFrame(data=d)

    if reviewFilePath is not None:
        try:
            with open(reviewFilePath, 'a') as review:
                review.write('Table of correspondences:\n')
                review.write('\n')
                review.write(df.to_string() + '\n')
                review.write('\n')
                review.close()
        finally:
            review.close()
    else:
        print(df)
    return result


def rectify_preds(Y_pred, Y_true, motif, reviewFilePath):
    rectified_preds = []
    classes_name = ['area', 'bar', 'barcode', 'boxplot', 'bubble', 'column', 'diverging bar',
                    'diverging stacked bar', 'donut', 'dot strip plot', 'heatmap', 'line',
                    'line column', 'lollipop', 'ordered bar', 'ordered column', 'paired bar',
                    'paired column', 'pie', 'population pyramid', 'proportional stacked bar',
                    'scatter', 'spine', 'stacked column', 'violin plot']

    for pred, true_class in zip(Y_pred, Y_true):
        pred = convert_in_lowercase(pred)

        if motif is not None:
            index = pred.find(motif)
            # If 'motif' is found in the chart_type, we cut the class up to this index
            if index != -1:
                pred = pred[:index]
        
        if pred == "bar graph":
            pred = "bar"
        elif pred == "diverging":
            pred = "diverging bar"
        elif pred == "circle":
            pred = "donut"
        elif pred == "histogram":
            pred = "column"
        elif pred == "line graph":
            pred = "line"
        elif pred == "violinplot":
            pred = "violin plot"
        elif pred == "pyramid":
            pred = "population pyramid"
        elif (pred == "heat map" or pred == "headmap" or pred == "hexmap" or pred == "hestmap"):
            pred = "heatmap"
        elif (pred == "box plot" or pred == "box and whisker plot"):
            pred = "boxplot"
        elif (pred == "strip" or pred == "strip plot"):
            pred = "dot strip plot"
        elif pred == "clustered stacked column":
            pred = "stacked column"
        elif (pred == "stacked bar" and true_class == "diverging stacked bar"):
            pred = "diverging stacked bar"
        elif (pred == "stacked bar" and true_class == "stacked column"):
            pred = "stacked column"
        elif (pred == "stacked bar" and true_class == "proportional stacked bar"):
            pred = "proportional stacked bar"
        elif (pred == "interval density bar" or
              pred == "interval bar" or
              pred == "interval scaled bar"):
            pred = "proportional stacked bar"
        elif pred == "scatter plot":
            pred = "scatter"
        elif "barcode" in pred:
            pred = "barcode"
        elif "dot strip plot" in pred:
            pred = "dot strip plot"
        
        rectified_preds.append(pred)
    
    extern_cl = detect_external_classes(rectified_preds, classes_name, reviewFilePath)
    if len(extern_cl) > 0:
        rectified_preds = replace_external_classes_by_other(rectified_preds, Y_true, extern_cl, reviewFilePath)
    
    return rectified_preds


def run_experiment(model_path, img_array_list, prompt_type):
    reviewFilePath = 'review/llava-v1.5-7b/Review_prompt_' + prompt_type + ".txt"
    prompt = None

    if prompt_type == "first":
        prompt = """USER:
                What is the chart type? Answer by just giving the chart type.\n
                ASSISTANT:"""
    elif prompt_type == "second":
        prompt = """USER:
                What is the chart type among the types in the list below:\n
                [area, bar, barcode, boxplot, bubble, column, diverging bar, diverging stacked bar, donut, dot strip plot, heatmap, line, line column, lollipop, ordered bar, ordered column, paired bar, paired column, pie, population pyramid, proportional stacked bar, scatter, spine, stacked column, violin plot]
                \nAnswer by giving just the best chart type in the previous list.\n
                ASSISTANT:"""
    elif prompt_type == "third":
        prompt = """USER:
                After analyzing the chart, classify it correctly into one of the following chart types:\n
                area, bar, barcode, boxplot, bubble, column, diverging bar, diverging stacked bar, donut, dot strip plot, heatmap, line, line column, lollipop, ordered bar, ordered column, paired bar, paired column, pie, population pyramid, proportional stacked bar, scatter, spine, stacked column, violin plot.\n
                \nAfter that, give me just the correct chart type.\n
                ASSISTANT:"""
    elif prompt_type == "fourth":
        prompt = """USER:
                Let's imagine that you are a data visualization expert assistant. You know and master the chart types described below.\n
                You must help the user to classify a chart into one of the following types:\n
                - Area: A graph that displays quantitative data using filled areas under a line, showing the magnitude of changes over time.\n
                - Bar: A chart that uses rectangular bars to compare different categories of data, with the length of each bar representing the value.\n
                - Barcode: A graphical representation of data using vertical bars of varying widths, often used for scanning and identification.\n
                - Boxplot: A graph that displays the distribution of a dataset through its quartiles, highlighting the median, interquartile range, and potential outliers.\n
                - Bubble: A type of scatter plot where data points are replaced with bubbles, with the size of each bubble representing a third variable.\n
                - Column: Similar to a bar chart, but with vertical bars, used to compare different categories of data.\n
                - Diverging Bar: A bar chart where bars extend in opposite directions from a central point, used to show deviations from a baseline.\n
                - Diverging Stacked Bar: A stacked bar chart where segments extend in opposite directions from a central axis, often used to compare parts of a whole.\n
                - Donut: A variation of a pie chart with a hole in the center, used to show parts of a whole in a circular format.\n
                - Dot Strip Plot: A chart that uses dots to represent individual data points, often used to show distribution along a single axis.\n
                - Heatmap: A graphical representation of data where values are depicted by color, often used to show the intensity of data points in a matrix.\n
                - Line: A chart that uses lines to connect data points, showing trends or changes over time.\n
                - Line Column: A combination chart that uses both lines and columns to represent different data series in the same graph.\n
                - Lollipop: A variation of a bar chart where bars are replaced with lines ending in dots, used to highlight data points.\n
                - Ordered Bar: A bar chart with bars sorted in a specific order, usually from highest to lowest value.\n
                - Ordered Column: A column chart with columns sorted in a specific order, typically from highest to lowest value.\n
                - Paired Bar: A chart that compares two sets of data side by side using bars, often used for direct comparison between categories.\n
                - Paired Column: A column chart that compares two sets of data side by side, facilitating comparison between categories.\n
                - Pie: A circular chart divided into sectors, each representing a proportion of the whole.\n
                - Population Pyramid: A type of bar chart that displays population distribution by age and gender, with bars extending in opposite directions.\n
                - Proportional Stacked Bar: A stacked bar chart where segments are proportional to the total, used to compare parts of a whole across categories.\n
                - Scatter: A graph that shows the relationship between two variables using dots, each representing a data point.\n
                - Spine: A variant of a pie chart where segments are arranged along a spine, used to show proportions in a linear format.\n
                - Stacked Column: A column chart where columns are divided into segments, each representing a part of the whole.\n
                - Violin Plot: A combination of a boxplot and a density plot, showing the distribution of a dataset through its probability density.\n
                what is the chart type?\n
                ASSISTANT:"""
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,
                                                                           model_base=None,
                                                                           model_name=model_name,
                                                                           device="cuda")
    preds_list = []
    counter = 0
    startTime = time.time()

    for img_array in img_array_list:
        counter += 1
        if counter % 100 == 0:
            print(str(counter) + " th images !")
        
        if img_array.shape[2] == 4: # Check if the image contains an alpha channel, and if so, remove it
            img_array = img_array[:, :, :3]
        
        img_array_scaled = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array_scaled, "RGB")

        pred = custom_inference([img], prompt, model_path, model, tokenizer,
                                image_processor, temperature=0.2, max_new_tokens=40)
        preds_list.append(pred)
    
    runTime = time.time() - startTime
    roundRunTime = str(dt.timedelta(seconds=runTime))
    try:
        with open(reviewFilePath, 'w') as review:
            review.write('*******************************************************\n')
            review.write("             llava-v1.5-7b inference \n")
            review.write('*******************************************************\n')
            review.write('prompt = ' + prompt + '\n')
            review.write('\n')
            review.write('------------- Runtime -------------\n')
            review.write('Runtime = ' + str(runTime) + ' secondes, i.e: ' + str(roundRunTime) + '\n')
            review.write('\n')
            review.close()
    finally:
        review.close()
    
    return preds_list, prompt_type, reviewFilePath


if __name__ == '__main__':
    model_path = 'liuhaotian/llava-v1.5-7b'
    model_name = 'llava-v1.5-7b'

    classes_name = ['area', 'bar', 'barcode', 'boxplot', 'bubble', 'column', 'diverging bar',
                    'diverging stacked bar', 'donut', 'dot strip plot', 'heatmap', 'line',
                    'line column', 'lollipop', 'ordered bar', 'ordered column', 'paired bar',
                    'paired column', 'pie', 'population pyramid', 'proportional stacked bar',
                    'scatter', 'spine', 'stacked column', 'violin plot']
    
    dict_input_X_test = np.load('../../Image_Dataset/shuffled_input_X_test.npz')
    X_test = dict_input_X_test['arr_0']
    img_array_list = X_test

    dict_target_Y_test = np.load('../../Image_Dataset/shuffled_target_Y_test.npz')
    Y_test = dict_target_Y_test['arr_0']
    Y_label_test = mcm.Y_array_to_class_label_array(Y_test)

    classes_name.append("other")

    for pr in ["first", "second", "third", "fourth"]: # for each form of prompt
        preds_list, prompt_type, reviewFilePath = run_experiment(model_path, img_array_list, prompt_type=pr)
        corrected_preds = rectify_preds(preds_list, Y_label_test, " chart", reviewFilePath)
        mcm.get_all_results(Y_label_test, corrected_preds, classes_name, prompt_type, model_name, reviewFilePath)

