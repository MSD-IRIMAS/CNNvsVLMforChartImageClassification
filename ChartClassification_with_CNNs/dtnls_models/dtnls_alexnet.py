# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:09:27 2020

@author: Bruno
"""

import keras
from keras.models import Sequential
from keras.layers import (Dense, Conv2D, MaxPool2D, Flatten,
                          Dropout, BatchNormalization)

def dtnls_alexnet(classes_nb):
    model = Sequential()
    
    # C1
    model.add(Conv2D(filters=96,
                     input_shape=(224,224,3),
                     kernel_size=(11,11),
                     strides=(4,4),
                     padding="valid",
                     activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))
    
    # C2
    model.add(Conv2D(filters=256,
                     kernel_size=(5,5),
                     strides=(1,1),
                     padding="valid",
                     activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))
    
    # C3
    model.add(Conv2D(filters=384,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding="valid",
                     activation = "relu"))
    
    # C4
    model.add(Conv2D(filters=384,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding="valid",
                     activation = "relu"))
    
    # C5
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding="valid",
                     activation = "relu"))
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))
    
    # Fully Connected layer
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=classes_nb, activation="softmax"))
    
    return model

