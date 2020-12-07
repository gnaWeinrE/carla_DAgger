
from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten
from keras.optimizers import Adam

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


# class Agent:
#     def __init__(self, obs_dim, act_dim):
#         with tf.variable_scope("DAgger"):
#             self.inputs = tf.placeholder(tf.float32, shape=[None, obs_dim])
#             self.actions = tf.placeholder(tf.float32, shape=[None, act_dim])
#
#             h1 = tf.layers.dense(inputs=self.inputs, units=128, activation=tf.nn.relu)
#             h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
#             h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
#             self.pred_actions = tf.layers.dense(inputs=h3, units=act_dim, activation=None)
#
#             self._loss = tf.reduce_mean(tf.square(self.pred_actions - self.actions))
#             self._train = tf.train.AdamOptimizer().minimize(self._loss)
#
#     def train(self, sess, obs, acts):
#         sess.run(self._train, feed_dict={self.inputs: obs, self.actions: acts})
#
#     def predict(self, sess, obs):
#         return sess.run(self.pred_actions, feed_dict={self.inputs: obs})
#
#     def loss(self, sess, obs, acts):
#         return sess.run(self._loss, feed_dict={self.inputs: obs, self.actions: acts})


def model_base_5_residual_CNN(input_shape):
    input = Input(shape=input_shape)

    cnn_1 = Conv2D(64, (7, 7), padding='same')(input)
    cnn_1a = Activation('relu')(cnn_1)
    cnn_1c = Concatenate()([cnn_1a, input])
    cnn_1ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1c)

    cnn_2 = Conv2D(64, (5, 5), padding='same')(cnn_1ap)
    cnn_2a = Activation('relu')(cnn_2)
    cnn_2c = Concatenate()([cnn_2a, cnn_1ap])
    cnn_2ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2c)

    cnn_3 = Conv2D(128, (5, 5), padding='same')(cnn_2ap)
    cnn_3a = Activation('relu')(cnn_3)
    cnn_3c = Concatenate()([cnn_3a, cnn_2ap])
    cnn_3ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_3c)

    cnn_4 = Conv2D(256, (5, 5), padding='same')(cnn_3ap)
    cnn_4a = Activation('relu')(cnn_4)
    cnn_4c = Concatenate()([cnn_4a, cnn_3ap])
    cnn_4ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_4c)

    cnn_5 = Conv2D(512, (3, 3), padding='same')(cnn_4ap)
    cnn_5a = Activation('relu')(cnn_5)
    #cnn_5c = Concatenate()([cnn_5a, cnn_4ap])
    cnn_5ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_5a)

    flatten = Flatten()(cnn_5ap)

    return input, flatten


def model_head_hidden_dense(model_input, model_output, outputs, model_settings):

    # Main input (image)
    inputs = [model_input]

    x = model_output

    # Add additional inputs with more data and concatenate
    if 'kmh' in ['kmh']:
        kmh_input = Input(shape=(1,), name='kmh_input')
        x = Concatenate()([x, kmh_input])
        inputs.append(kmh_input)

    # Add additional fully-connected layer
    x = Dense(model_settings['hidden_1_units'], activation='relu')(x)

    # And finally output (regression) layer
    predictions = Dense(outputs, activation='linear')(x)

    # Create a model
    model = Model(inputs=inputs, outputs=predictions)

    return model


def create_model(prediction=False):

    # Create the model
    model_base = model_base_5_residual_CNN((160, 80, 3))
    model = model_head_hidden_dense(*model_base, outputs=3, model_settings={'hidden_1_units': 256})

    # self._extract_model_info(model)

    # We need to compile model only for training purposes, agents do not compile their models
    if not prediction:
        # self.compile_model(model=model, lr=0.001, decay=0.0)
        model.compile(loss="mse", optimizer=Adam(lr=0.001, decay=0.0), metrics=['accuracy'])
    # If we show convcam, we need additional output from given layer
    # elif self.show_conv_cam:
    #     model = Model(model.input, [model.output, model.layers[self.convcam_layer].output])

    return model

