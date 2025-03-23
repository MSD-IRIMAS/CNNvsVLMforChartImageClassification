import os
import keras
import numpy as np
import tensorflow as tf
import cnns_training as cnn_train


if __name__ == '__main__':
    # Reproducibility in models
    keras.utils.set_random_seed(812)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configure TensorFlow to allocate memory_limit Giga of memory to the 1st GPU
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=80000)]) # on Nvidia A100: 80000
            logical_gpus = tf.config.list_logical_devices('GPU')

            # activation of progressive GPU memory growth
            # memory growth must be the same on all GPUs
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            # logical_gpus = tf.config.list_logical_devices('GPU')

            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


    dict_input_X_train = np.load('../Image_Dataset/shuffled_input_X_train.npz')
    input_X_train = dict_input_X_train['arr_0']
    print('input_X_train loaded !')
    dict_target_Y_train = np.load('../Image_Dataset/shuffled_target_Y_train.npz')
    target_Y_train = dict_target_Y_train['arr_0']
    print('target_Y_train loaded !')
    dict_input_X_test = np.load('../Image_Dataset/shuffled_input_X_test.npz')
    input_X_test = dict_input_X_test['arr_0']
    print('input_X_test loaded !')
    dict_target_Y_test = np.load('../Image_Dataset/shuffled_target_Y_test.npz')
    target_Y_test = dict_target_Y_test['arr_0']
    print('target_Y_test loaded !')
    dict_input_X_validation = np.load('../Image_Dataset/shuffled_input_X_validation.npz')
    input_X_validation = dict_input_X_validation['arr_0']
    print('input_X_validation loaded !')
    dict_target_Y_validation = np.load('../Image_Dataset/shuffled_target_Y_validation.npz')
    target_Y_validation = dict_target_Y_validation['arr_0']
    print('target_Y_validation loaded !')

    classes_name = ['area', 'bar', 'barcode', 'boxplot', 'bubble', 'column', 'diverging bar',
                    'diverging stacked bar', 'donut', 'dot strip plot', 'heatmap', 'line',
                    'line column', 'lollipop', 'ordered bar', 'ordered column', 'paired bar',
                    'paired column', 'pie', 'population pyramid', 'proportional stacked bar',
                    'scatter', 'spine', 'stacked column', 'violin plot']
    
    # classes_nb = len(classes_name)
    # print(os.getcwd())
    # list_of_batch_divisors = cnn_train.get_batch_divisors_list(classes_nb, 1000, 0.2, 0.2)
    # print('Possible values of batch: ', list_of_batch_divisors)

    # Possible values of batch:
    # [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100, 125, 128, 160,
    #  200, 250, 320, 400, 500, 640, 800, 1000, 1600, 2000, 3200, 4000, 8000, 16000]
    
    training_type = 'full' # Possible values 'full' or 'fine-tuning'
    all_models = ['dtnls_alexnet', 'VGG16', 'InceptionV3', 'Xception', 'InceptionResNetV2', 'EfficientNetB4']
    batch_size = 64
    epochs = 100
    cnn_train.train_all_models(classes_name, training_type, all_models, batch_size, epochs,
                               input_X_train, target_Y_train, input_X_test, target_Y_test,
                               input_X_validation, target_Y_validation)
    
    # training_type = 'fine-tuning' # Possible values 'full' or 'fine-tuning'
    # pretrained_models = ['VGG16', 'InceptionV3', 'Xception', 'InceptionResNetV2', 'EfficientNetB4']
    # batch_size = 64
    # epochs = 100
    # cnn_train.train_all_models(classes_name, training_type, pretrained_models, batch_size, epochs,
    #                            input_X_train, target_Y_train, input_X_test, target_Y_test,
    #                            input_X_validation, target_Y_validation)

    # training_type = 'full' # # Possible values 'full' or 'fine-tuning'
    # model_name = 'dtnls_alexnet'
    # batch_size = 64
    # epochs = 100
    # cnn_train.train_single_model(classes_name, training_type, model_name, batch_size, epochs,
    #                              input_X_train, target_Y_train, input_X_test, target_Y_test,
    #                              input_X_validation, target_Y_validation)

