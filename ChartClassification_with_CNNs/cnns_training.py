import time
import datetime as dt
import os
import gc
# import sys
import keras
import numpy as np
from dtnls_models.dtnls_alexnet import dtnls_alexnet
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
from keras.models import load_model # type: ignore
from keras.models import Model # type: ignore
from keras.layers import Dense, GlobalAveragePooling2D # type: ignore
import tensorflow as tf
from plot_confusion_matrix import get_all_results
from plot_confusion_matrix import Y_array_to_class_label_array


def get_batch_divisors_list(classes_nb, img_nb_in_class, test_size, validation_size):
    list_of_batch_divisors = []
    total_nb_img = classes_nb * img_nb_in_class
    test_img_nb = total_nb_img * test_size
    train_val_img_nb = total_nb_img - test_img_nb
    validation_img_nb = train_val_img_nb * validation_size
    train_img_nb = train_val_img_nb - validation_img_nb
    
    for d in range(1, int(train_img_nb) + 1):
        if train_img_nb % d == 0:
            list_of_batch_divisors.append(d)
    
    return list_of_batch_divisors


def preprocess_data_for_model(model_name, shuffled_input_X, shuffled_target_Y):
    print('Preprocessing data for ' + model_name + ' !')
    input_X = []
    IMG_SIZE = 0
    if model_name == 'dtnls_alexnet':
        IMG_SIZE = 224
        for imgArray in shuffled_input_X:
            img = keras.utils.array_to_img(imgArray)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            x = keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            input_X.append(x)
    elif model_name == 'VGG16':
        IMG_SIZE = 224
        for imgArray in shuffled_input_X:
            img = keras.utils.array_to_img(imgArray)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            x = keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = keras.applications.vgg16.preprocess_input(x) # VGG16-specific preprocessing
            input_X.append(x)
    elif model_name == 'InceptionV3':
        IMG_SIZE = 299
        for imgArray in shuffled_input_X:
            img = keras.utils.array_to_img(imgArray)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            x = keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = keras.applications.inception_v3.preprocess_input(x) # InceptionV3-specific preprocessing
            input_X.append(x)
    elif model_name == 'Xception':
        IMG_SIZE = 299
        for imgArray in shuffled_input_X:
            img = keras.utils.array_to_img(imgArray)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            x = keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = keras.applications.xception.preprocess_input(x) # Xception-specific preprocessing
            input_X.append(x)
    elif model_name == 'InceptionResNetV2':
        IMG_SIZE = 299
        for imgArray in shuffled_input_X:
            img = keras.utils.array_to_img(imgArray)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            x = keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = keras.applications.inception_resnet_v2.preprocess_input(x) # InceptionResNetV2-specific preprocessing
            input_X.append(x)
    elif model_name == 'EfficientNetB4':
        IMG_SIZE = 380
        for imgArray in shuffled_input_X:
            img = keras.utils.array_to_img(imgArray)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            x = keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = keras.applications.efficientnet.preprocess_input(x) # EfficientNetB4-specific preprocessing
            input_X.append(x)
    
    input_X = np.array(input_X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    target_Y = keras.utils.to_categorical(shuffled_target_Y, num_classes=25)
    
    return input_X, target_Y


def set_model(training_type, model_name, classes_nb):
    include_top = False
    weights = None
    input_tensor = None
    input_shape = None
    pooling = None
    classes = None
    classifier_activation = None

    if training_type == "fine-tuning":
        print('Loading pretrained ' + model_name + ' !')
        include_top = False
        weights = 'imagenet'
        input_tensor = None
        input_shape = None
        pooling = None
        classes = None
        classifier_activation = None
    elif training_type == "full":
        print('Loading empty ' + model_name + ' !')
        include_top = True
        weights = None
        input_tensor = None
        input_shape = None
        pooling = None
        classes = classes_nb
        classifier_activation = "softmax"
    
    model = None
    if model_name == 'dtnls_alexnet':
        model = dtnls_alexnet(classes_nb)
    elif model_name == 'VGG16':
        model = keras.applications.vgg16.VGG16(include_top=include_top,
                                               weights=weights,
                                               input_tensor=input_tensor,
                                               input_shape=input_shape,
                                               pooling=pooling,
                                               classes=classes,
                                               classifier_activation=classifier_activation)
    elif model_name == 'InceptionV3':
        model = keras.applications.inception_v3.InceptionV3(include_top=include_top,
                                                            weights=weights,
                                                            input_tensor=input_tensor,
                                                            input_shape=input_shape,
                                                            pooling=pooling,
                                                            classes=classes,
                                                            classifier_activation=classifier_activation)
    elif model_name == 'Xception':
        model = keras.applications.xception.Xception(include_top=include_top,
                                                     weights=weights,
                                                     input_tensor=input_tensor,
                                                     input_shape=input_shape,
                                                     pooling=pooling,
                                                     classes=classes,
                                                     classifier_activation=classifier_activation)
    elif model_name == 'InceptionResNetV2':
        model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=include_top,
                                                                         weights=weights,
                                                                         input_tensor=input_tensor,
                                                                         input_shape=input_shape,
                                                                         pooling=pooling,
                                                                         classes=classes,
                                                                         classifier_activation=classifier_activation)
    elif model_name == 'EfficientNetB4':
        model = keras.applications.EfficientNetB4(include_top=include_top,
                                                  weights=weights,
                                                  input_tensor=input_tensor,
                                                  input_shape=input_shape,
                                                  pooling=pooling,
                                                  classes=classes,
                                                  classifier_activation=classifier_activation)
    return model


def show_layer_indice_and_name_in_model(model):
    for i, layer in enumerate(model.layers):
        print(i, layer.name)


def freeze_the_first_N_layers_in_model(model, N):
    for layer in model.layers[:N]:
        layer.trainable = False
    for layer in model.layers[N:]:
        layer.trainable = True
    return model


def get_model_classes_predictions(model, X_test):
    print('Class predictions computing !')
    preds = model.predict(X_test)
    preds_cl_list = []
    for pred in preds:
        pred_cl_index = np.argmax(pred, axis=0)
        preds_cl_list.append(pred_cl_index)
    
    Y_preds = np.array(preds_cl_list, dtype='int32')
    return Y_preds


def plot_train_valid_loss(training_type, model_name, batch_size, histo, dateAndTime):
    
    plt.figure(figsize=(12,8))
    plt.plot(histo.history['loss'], label='Training loss values')
    plt.plot(histo.history['val_loss'], label='Validation loss values')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss values')
    plt.legend(loc='best')
    startPath = 'saved_models/' + model_name + '_' + training_type
    endPath = '_'+str(batch_size)+'_loss_values_'+dateAndTime+'.png'
    savedLossFigPath = startPath + endPath
    plt.savefig(savedLossFigPath)
    plt.show()


def plot_train_valid_accuracy(training_type, model_name, batch_size, histo, dateAndTime):
    
    plt.figure(figsize=(12,8))
    plt.plot(histo.history['categorical_accuracy'],
             label='Training categorical accuracy values')
    plt.plot(histo.history['val_categorical_accuracy'],
             label='Validation categorical accuracy values')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical accuracy')
    plt.title('Categorical accuracy values')
    plt.legend(loc='best')
    startPath = 'saved_models/' + model_name + '_' + training_type + '_'
    endPath = str(batch_size)+'_categorical_accuracy_values_'+dateAndTime+'.png'
    savedAccFigPath = startPath + endPath
    plt.savefig(savedAccFigPath)
    plt.show()


def model_evaluation_and_review(training_type, savedModelCheckPath, model_name, classes_name,
                                batch_size, epochs, X_test, Y_test, Y_test_confMatrix,
                                dateAndTime, runTime, roundRunTime):
    
    model = load_model(savedModelCheckPath)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print('loss: {0}, categorical_accuracy: {1}'.format(scores[0],scores[1]))

    startPath = 'saved_models/Review_' + model_name + '_' + training_type
    endPath = '_' + str(batch_size) + '_'  + dateAndTime + '.txt'
    reviewFilePath = startPath + endPath
    try:
        with open(reviewFilePath, 'w') as review:
            review.write('*************************************************************************\n')
            review.write(model_name +' with batch size = '+ str(batch_size) +' and epochs = '+str(epochs)+'\n')
            review.write('*************************************************************************\n')
            review.write('training type: ' + training_type + '\n')
            review.write('\n')
            review.write('----- Evaluation on test dataset -----\n')
            review.write('loss: ' + str(scores[0]) + '\n')
            review.write('categorical_accuracy: ' + str(scores[1]) + '\n')
            review.write('\n')
            review.write('----- Training runtime -----\n')
            review.write('Runtime = ' + str(runTime) + ' secondes, i.e: ' + str(roundRunTime) + '\n')
            review.write('\n')
            review.write('********************************** END **********************************\n')
            review.close()
    finally:
        review.close()
    
    Y_preds = get_model_classes_predictions(model, X_test)
    beginPath = 'saved_models/Confusion_matrix_' + model_name + '_'+ training_type
    nextPath = '_'+str(batch_size)+'_'+dateAndTime+'.png'
    confMatPath = beginPath + nextPath
    # parent_folder = os.path.abspath('..')
    # sys.path.append(parent_folder)
    # from plot_confusion_matrix import matrice_confusion
    Y_label_test = Y_array_to_class_label_array(Y_test_confMatrix)
    Y_label_pred = Y_array_to_class_label_array(Y_preds)
    get_all_results(Y_label_test, Y_label_pred, classes_name, model_name, reviewFilePath, confMatPath)


def train_full_model(model, batch_size, epochs, X_train, Y_train, validation_data, modelCheckpoint):
    sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=1e-6)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    histo = model.fit(X_train, Y_train, validation_data=validation_data,
                      epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[modelCheckpoint])
    return histo


def fine_tune_model(model_name, model, classes_nb, batch_size, epochs,
                    X_train, Y_train, validation_data, modelCheckpoint, dateAndTime):
    x = model.output
    x = GlobalAveragePooling2D()(x) # Add an average pooling layer
    x = Dense(1024, activation='relu')(x) # Add a fully connected layer
    # Softmax layer for predicting chart classes
    predictions = Dense(classes_nb, activation='softmax')(x)
    # Model that we are going to train
    tuned_model = Model(inputs=model.input, outputs=predictions)

    # Freeze the layers of the original pre-trained model
    # for layer in model.layers:
    #     layer.trainable = False
    model.trainable = False
    
    sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=1e-6)
    tuned_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    print('Train the top layer of the tuned model !')
    histo = tuned_model.fit(X_train, Y_train, validation_data=validation_data,
                            epochs=40, batch_size=batch_size, verbose=1, callbacks=[es])
    
    model_top_layer = model_name + '_' + 'top_layer'
    plot_train_valid_loss('fine-tuning', model_top_layer, batch_size, histo, dateAndTime)
    plot_train_valid_accuracy('fine-tuning', model_top_layer, batch_size, histo, dateAndTime)
    
    # Unfreeze the layers of the original pre-trained model
    # for layer in model.layers:
    #     layer._trainable = True
    model.trainable = True
    
    sgd = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, weight_decay=1e-6) # Low learning rate
    tuned_model.compile(optimizer=sgd, # or keras.optimizers.Adam(1e-4)
                        loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    histo = tuned_model.fit(X_train, Y_train, validation_data=validation_data, epochs=epochs,
                            batch_size=batch_size, verbose=1, callbacks=[modelCheckpoint])
    return histo


def start_training(training_type, model_name, model, classes_name, batch_size, epochs,
                   X_train, Y_train, X_test, Y_test, validation_data, Y_test_confMatrix):
    fr_date = time.strftime("%d_%m_%Y")
    heure = time.strftime("%Hh%M")
    dateAndTime = fr_date + '_' + heure
    startPath = 'saved_models/' + model_name +'_'+ training_type
    endPath = '_' + str(batch_size) + '_' + dateAndTime +'.keras'
    savedModelCheckPath = startPath + endPath

    mc = ModelCheckpoint(savedModelCheckPath, monitor='val_loss', mode='min',
                         verbose=1, save_weights_only=False, save_best_only=True)
    histo = None
    startTime = time.time()
    if training_type == 'full':
        print('Start ' + model_name + ' full training !')
        histo = train_full_model(model, batch_size, epochs, X_train, Y_train, validation_data, mc)
    elif training_type == 'fine-tuning':
        print('Start ' + model_name + ' fine-tuning !')
        classes_nb = len(classes_name)
        histo = fine_tune_model(model_name, model, classes_nb, batch_size, epochs,
                                X_train, Y_train, validation_data, mc, dateAndTime)
    
    plot_train_valid_loss(training_type, model_name, batch_size, histo, dateAndTime)
    plot_train_valid_accuracy(training_type, model_name, batch_size, histo, dateAndTime)

    runTime = time.time() - startTime
    roundRunTime = str(dt.timedelta(seconds=runTime))
    print('Runtime = ', runTime, ' secondes, i.e: ', roundRunTime)

    model_evaluation_and_review(training_type, savedModelCheckPath, model_name, classes_name,
                                batch_size, epochs, X_test, Y_test, Y_test_confMatrix,
                                dateAndTime, runTime, roundRunTime)



def train_single_model(classes_name, training_type, model_name, batch_size, epochs,
                       input_X_train, target_Y_train, input_X_test, target_Y_test,
                       input_X_validation, target_Y_validation):
    
    X_train, Y_train = preprocess_data_for_model(model_name, input_X_train, target_Y_train)
    print('X_train shape: {0}, Y_train shape: {1}'.format(X_train.shape, Y_train.shape))
    X_test, Y_test = preprocess_data_for_model(model_name, input_X_test, target_Y_test)
    print('X_test shape: {0}, Y_test shape: {1}'.format(X_test.shape, Y_test.shape))
    X_valida, Y_valida = preprocess_data_for_model(model_name, input_X_validation, target_Y_validation)
    print('X_valida shape: {0}, Y_valida shape: {1}'.format(X_valida.shape, Y_valida.shape))
    validation_data = (X_valida, Y_valida)

    classes_nb = Y_train.shape[1]
    print('Chart classes number: ', classes_nb)
    model = set_model(training_type, model_name, classes_nb)

    start_training(training_type, model_name, model, classes_name, batch_size, epochs,
                   X_train, Y_train, X_test, Y_test, validation_data, target_Y_test)
    
    del model, X_train, Y_train, X_test, Y_test, validation_data, X_valida, Y_valida
    time.sleep(20)
    gc.collect()
    keras.backend.clear_session()


def eval_single_model(input_X_test, target_Y_test, training_type, savedModelCheckPath,
                      model_name, classes_name, batch_size, epochs, runTime, roundRunTime):
    
    fr_date = time.strftime("%d_%m_%Y")
    heure = time.strftime("%Hh%M")
    dateAndTime = fr_date + '_' + heure
    
    X_test, Y_test = preprocess_data_for_model(model_name, input_X_test, target_Y_test)
    print('X_test shape: {0}, Y_test shape: {1}'.format(X_test.shape, Y_test.shape))

    model_evaluation_and_review(training_type, savedModelCheckPath, model_name, classes_name,
                                batch_size, epochs, X_test, Y_test, target_Y_test,
                                dateAndTime, runTime, roundRunTime)


def train_all_models(classes_name, training_type, model_name_list, batch_size, epochs,
                     input_X_train, target_Y_train, input_X_test, target_Y_test,
                     input_X_validation, target_Y_validation):
    
    for model_name in model_name_list:
        X_train, Y_train = preprocess_data_for_model(model_name, input_X_train, target_Y_train)
        print('X_train shape: {0}, Y_train shape: {1}'.format(X_train.shape, Y_train.shape))
        X_test, Y_test = preprocess_data_for_model(model_name, input_X_test, target_Y_test)
        print('X_test shape: {0}, Y_test shape: {1}'.format(X_test.shape, Y_test.shape))
        X_valida, Y_valida = preprocess_data_for_model(model_name, input_X_validation, target_Y_validation)
        print('X_valida shape: {0}, Y_valida shape: {1}'.format(X_valida.shape, Y_valida.shape))
        validation_data = (X_valida, Y_valida)

        classes_nb = Y_train.shape[1]
        print('Chart classes number: ', classes_nb)
        model = set_model(training_type, model_name, classes_nb)

        start_training(training_type, model_name, model, classes_name, batch_size, epochs,
                       X_train, Y_train, X_test, Y_test, validation_data, target_Y_test)
        
        del model, X_train, Y_train, X_test, Y_test, validation_data, X_valida, Y_valida
        time.sleep(20)
        gc.collect()
        keras.backend.clear_session()


