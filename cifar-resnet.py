import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Input, PReLU, ReLU, Flatten,
                          Activation, Add, ZeroPadding2D, Conv2D,
                          GlobalAveragePooling2D, Dense,
                          MaxPooling2D, BatchNormalization)
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Model, Sequential
from keras.utils import plot_model, to_categorical
from IPython.display import Image

def build_cifar_resnet(class_num=10, input_shape=(32, 32, 3)):
    model_input = Input(shape=input_shape)

    X = Conv2D(64, kernel_size=(3, 3), padding='same')(model_input)
    X = BatchNormalization()(X)
    X = PReLU()(X)

    B2 = Conv2D(64, kernel_size=(3, 3), padding='same')(X)
    B2 = BatchNormalization()(B2)
    B2 = PReLU()(B2)
    B2 = Conv2D(64, kernel_size=(3, 3), padding='same')(B2)
    B2 = BatchNormalization()(B2)
    X = Add()([X, B2])
    X = PReLU()(X)

    X = ZeroPadding2D(padding=(1, 1))(X)
    B1 = Conv2D(128, kernel_size=(3, 3), strides=2)(X)
    B1 = BatchNormalization()(B1)
    B2 = Conv2D(128, kernel_size=(3, 3), strides=2)(X)
    B2 = BatchNormalization()(B2)
    B2 = PReLU()(B2)
    B2 = Conv2D(128, kernel_size=(3, 3), padding='same')(B2)
    B2 = BatchNormalization()(B2)
    X = Add()([B1, B2])
    X = PReLU()(X)

    X = ZeroPadding2D(padding=(1, 1))(X)
    B1 = Conv2D(256, kernel_size=(3, 3), strides=2)(X)
    B1 = BatchNormalization()(B1)
    B2 = Conv2D(256, kernel_size=(3, 3), strides=2)(X)
    B2 = BatchNormalization()(B2)
    B2 = PReLU()(B2)
    B2 = Conv2D(256, kernel_size=(3, 3), padding='same')(B2)
    B2 = BatchNormalization()(B2)
    X = Add()([B1, B2])
    X = PReLU()(X)

    X = GlobalAveragePooling2D()(X)
    X = Dense(class_num, activation='softmax')(X)


    model = Model(inputs=model_input, outputs=X)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

cifar_resnet = build_cifar_resnet()
cifar_resnet.summary()

