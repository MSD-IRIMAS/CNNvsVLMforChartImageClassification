import glob
import gc
import time
import keras
import numpy as np
from random import shuffle


def shuffle_supervised_data(imgByClass_db):
    shuffled_supervised_data = []
    for chartClass in imgByClass_db:
        for img in chartClass:
            label = imgByClass_db.index(chartClass)
            img = keras.utils.load_img(img, target_size=(400, 400))
            x = keras.utils.img_to_array(img)
            shuffled_supervised_data.append((x, label))
    
    shuffle(shuffled_supervised_data)

    input_X = np.array([couple[0] for couple in shuffled_supervised_data])
    target_Y = np.array([couple[1] for couple in shuffled_supervised_data], dtype='int32')
    return input_X, target_Y


if __name__ == '__main__':
    classes_list = ['area', 'bar', 'barcode_plot', 'boxplot', 'bubble', 'column',
                    'diverging_bar', 'diverging_stacked_bar', 'donut', 'dot_strip_plot',
                    'heatmap', 'line', 'line_column', 'lollipop', 'ordered_bar',
                    'ordered_column', 'paired_bar', 'paired_column', 'pie',
                    'population_pyramid', 'proportional_stacked_bar', 'scatter_plot',
                    'spine', 'stacked_column', 'violin_plot']
    
    imgByClassForTest_db = []
    imgByClassForTrain_db = []
    imgByClassForValidation_db = []

    for className in classes_list:
        imgByClassForTest = glob.glob('chartClasses/test/' + className + '/*')
        imgByClassForTest_db.append(imgByClassForTest)
        imgByClassForTrain = glob.glob('chartClasses/train/' + className + '/*')
        imgByClassForTrain_db.append(imgByClassForTrain)
        imgByClassForValidation = glob.glob('chartClasses/validation/' + className + '/*')
        imgByClassForValidation_db.append(imgByClassForValidation)


    input_X_test, target_Y_test = shuffle_supervised_data(imgByClassForTest_db)
    del imgByClassForTest_db
    time.sleep(5)
    gc.collect()
    print('imgByClassForTest_db DELETED !')
    np.savez_compressed('shuffled_input_X_test.npz', input_X_test)
    del input_X_test
    time.sleep(5)
    gc.collect()
    print('input_X_test DELETED !')
    np.savez_compressed('shuffled_target_Y_test.npz', target_Y_test)
    del target_Y_test
    time.sleep(5)
    gc.collect()
    print('target_Y_test DELETED !')
    input_X_train, target_Y_train = shuffle_supervised_data(imgByClassForTrain_db)
    del imgByClassForTrain_db
    time.sleep(10)
    gc.collect()
    print('imgByClassForTrain_db DELETED !')
    np.savez_compressed('shuffled_input_X_train.npz', input_X_train)
    del input_X_train
    time.sleep(10)
    gc.collect()
    print('input_X_train DELETED !')
    np.savez_compressed('shuffled_target_Y_train.npz', target_Y_train)
    del target_Y_train
    time.sleep(10)
    gc.collect()
    print('target_Y_train DELETED !')
    input_X_validation, target_Y_validation = shuffle_supervised_data(imgByClassForValidation_db)
    del imgByClassForValidation_db
    time.sleep(5)
    gc.collect()
    print('imgByClassForValidation_db DELETED !')
    np.savez_compressed('shuffled_input_X_validation.npz', input_X_validation)
    del input_X_validation
    time.sleep(5)
    gc.collect()
    print('input_X_validation DELETED !')
    np.savez_compressed('shuffled_target_Y_validation.npz', target_Y_validation)
    del target_Y_validation
    time.sleep(5)
    gc.collect()
    print('target_Y_validation DELETED !')

