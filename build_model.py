from tensorflow.keras import layers, Model
import tensorflow as tf
from typing import Tuple


def build_model(polyline_output_shape: int, label_output_shape: int, input_shape: Tuple[int, int, int]):
    model_name = 'lane_line_cnn_model'
    # pretrained
    pre_trained_model = tf.keras.applications.InceptionResNetV2(input_shape=input_shape,
                                                                weights='imagenet',
                                                                include_top=False)
    global_max_pool = layers.GlobalMaxPool2D()(pre_trained_model.output)
    dropout_max_pool = layers.Dropout(.2)(global_max_pool)

    # polyline part
    dense_polyline = tf.keras.layers.Dense(units=512, activation='relu')(dropout_max_pool)
    batch_norm = layers.BatchNormalization()(dense_polyline)
    dropout_polyine = layers.Dropout(.2)(batch_norm)

    dense_polyline_2 = tf.keras.layers.Dense(units=512, activation='relu')(dropout_polyine)
    batch_norm2 = layers.BatchNormalization()(dense_polyline_2)
    dropout_polyine_2 = layers.Dropout(.2)(batch_norm2)

    # label common part
    dense_label = tf.keras.layers.Dense(units=256, activation='relu')(dropout_max_pool)
    batch_norm = layers.BatchNormalization()(dense_label)
    dropout_label = layers.Dropout(.2)(batch_norm)

    # lane 1 part
    x = tf.keras.layers.Dense(units=128, activation='relu')(dropout_label)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(.2)(x)

    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(.2)(x)

    x = tf.keras.layers.Dense(units=32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(.2)(x)

    # lane 2 part
    y = tf.keras.layers.Dense(units=128, activation='relu')(dropout_label)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(.2)(y)

    y = tf.keras.layers.Dense(units=64, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(.2)(y)

    y = tf.keras.layers.Dense(units=32, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(.2)(y)

    # output
    polyline_output = layers.Dense(polyline_output_shape, name='polyline_output')(dropout_polyine_2)
    label_output_1 = layers.Dense(label_output_shape, activation='softmax', name='label_output_1')(x)
    label_output_2 = layers.Dense(label_output_shape, activation='softmax', name='label_output_2')(y)

    model = Model(pre_trained_model.input, outputs=[
        polyline_output,
        label_output_1,
        label_output_2,
    ], name=model_name
                  )

    return model, pre_trained_model
