import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf 
from matplotlib import pyplot as plt
from PIL import Image
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Lambda, Dense, Flatten, Input
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model


def feature_extract( inputs):
    l1 = inputs
    l2 = Conv2D(64, (3,3), padding='same', activation='relu')(l1)
    l3 = Conv2D(64, (3,3), padding='same', activation='relu')(l2)
    l4 = MaxPooling2D((2,2), strides=(2,2))(l3)
    l5 = Conv2D(128, (3,3), padding='same', activation='relu')(l4)
    l6 = Conv2D(128, (3,3), padding='same', activation='relu')(l5)
    l7 = MaxPooling2D((2,2), strides=(2,2))(l6)
    l8 = Conv2D(256, (3,3), padding='same', activation='relu')(l7)
    l9 = Conv2D(256, (3,3), padding='same', activation='relu')(l8)
    l10 = Conv2D(256, (3,3), padding='same', activation='relu')(l9)
    l11 = MaxPooling2D((2,2), strides=(2,2))(l10)
    l12 = Conv2D(512, (3,3), padding='same', activation='relu')(l11)
    l13 = Conv2D(512, (3,3), padding='same', activation='relu')(l12)
    l15 = Conv2D(512, (3,3), padding='same', activation='relu')(l13)
    l16 = MaxPooling2D((2,2), strides=(2,2))(l15)
    l17 = Conv2D(512, (3,3), padding='same', activation='relu')(l16)
    l18 = Conv2D(512, (3,3), padding='same', activation='relu')(l17)
    l19 = Conv2D(512, (3,3), padding='same', activation='relu')(l18)
    l20 = MaxPooling2D((2,2), strides=(2,2))(l19)
    return l20

def age_branch( inputs):
    l1 = tf.keras.layers.Flatten()(inputs)
    l2 = tf.keras.layers.Dense(256, activation='relu')(l1)
    l3 = tf.keras.layers.Dense(256, activation='relu')(l2)
    l4 = tf.keras.layers.Dense(1, activation='linear', name="age_output")(l3)
    return l4

def gender_branch( inputs, num_genders=2):
    l1 = tf.keras.layers.Flatten()(inputs)
    l2 = tf.keras.layers.Dense(128, activation='relu')(l1)
    l3 = tf.keras.layers.Dense(64, activation='relu')(l2)
    l4 = tf.keras.layers.Dense(num_genders, activation='softmax', name="gender_output")(l3)
    return l4

def race_branch( inputs, num_races):
    l1 = tf.keras.layers.Flatten()(inputs)
    l2 = tf.keras.layers.Dense(128, activation='relu')(l1)
    l3 = tf.keras.layers.Dense(64, activation='relu')(l2)
    l4 = tf.keras.layers.Dense(num_races, activation='softmax', name="race_output")(l3)
    return l4

def full_model( width, height, num_races):
    input_shape = (height, width, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    feature_extractor = feature_extract(inputs)
    age_output = age_branch(feature_extractor)
    gender_output = gender_branch(feature_extractor)
    race_output = race_branch(feature_extractor, num_races)
    model = tf.keras.Model(inputs=inputs,
                  outputs=[age_output, race_output, gender_output],
                  name="utk_face_net")
    return model