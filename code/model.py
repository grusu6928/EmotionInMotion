from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math



#“Conv(kernel size)-[stride size]-[number of channels]”. The maxpooling layer parameters are denoted as “Maxpool-[kernel
#size]-[stride size]”. For brevity, the local response normalization layer and
#ReLU activation function is not shown
class Model(tf.keras.Model):
    def __init__(self, f, t):
        super(Model, self).__init__()
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.batch_size = 24
        # num channels,kernel size, stride size
        self.layer1 = tf.keras.layers.Conv1D(96, 11, strides = 4, activation = "relu")
        self.layer2 = tf.keras.layers.Conv1D(256, 5, strides = 1, activation = "relu")
        self.layer3 = tf.keras.layers.Conv1D(384, 3, strides = 1, activation = "relu")
        self.layer4 = tf.keras.layers.Conv1D(384, 3, strides = 1, activation = "relu")
        self.layer5 = tf.keras.layers.Conv1D(256, 3, strides = 1, activation = "relu") #!!!!!!!!!!
        #“Maxpool-[kernelsize]-[stride size]”
        self.max1 = tf.keras.layers.MaxPool1D(pool_size = 3, strides = 2)
        self.max2 = tf.keras.layers.MaxPool1D(pool_size = 3, strides = 2)
        self.max3 = tf.keras.layers.MaxPool1D(pool_size = 3, strides = 2)

        #attention stuff
        self.u = tf.Variable(tf.random.truncated_normal((672, 1)))
        self.linear_layer = tf.keras.layers.Dense((672), activation="tanh", dtype='float32') #might be different f and t
        self.test_layer = tf.keras.layers.Dense((4), activation="tanh", dtype='float32'
        self.lammda = 0.3

    def FCN_layer(self, speech_spectrogram):
        #output of FCN encoder is a 3-dimensional
        # array of size F × T × C,where the F and T correspond to
        # the frequency and time domains of spectrogram and C is
        # channel size
        # lrn should take in 4d tensor as first input, but our conv2d uses 3d tensors, consider expand_dims or something else
        #TODO: lrn is commented out of now
        #layer1out = tf.nn.lrn(self.layer1(speech_spectrogram))
        layer1out = self.max1(speech_spectrogram)
        #layer1out = tf.nn.lrn(self.layer2(layer1out))
        layer2out = self.max2(layer1out)
        layer3out = self.layer3(layer2out)
        layer4out = self.layer4(layer3out)
        layer5out = self.layer5(layer4out)
        FCN_out = self.max3(layer5out)


        return FCN_out

    def attention_layer(self, fcn_output):
        #print("fcn_output shape")
        #print(fcn_output.shape)
        F = fcn_output.shape[0]
        T = fcn_output.shape[1]
        C = fcn_output.shape[2]
        L = F*T
        A = tf.reshape(fcn_output, (L, C))
        e_tensor = tf.matmul(tf.transpose(self.u), self.linear_layer(A))
        print("e_tensor shape")
        print(e_tensor.shape)
        exp_tensor = tf.math.exp(self.lammda * e_tensor)
        print("exp_tensor shape")
        print(exp_tensor.shape)
        alpha_tensor = exp_tensor / tf.reduce_sum(exp_tensor)
        print("alpha_tensor shape")
        print(alpha_tensor.shape)
        alpha_times_A = tf.transpose(alpha_tensor) * A
        print(alpha_times_A.shape)
        c = tf.reduce_sum(alpha_times_A, axis = 0) #might need an axis
        print(c.shape) #shape is 256
        return c

    def call(self, spectogram):
        fcn_output = self.FCN_layer(spectogram)
        print(fcn_output.shape)
        print("fcn finished!")
        #attention_output = self.attention_layer(fcn_output)
        self.test_layer(tf.flatten(fcn_output))
        #softmax
        return tf.nn.softmax(attention_output)

    def loss(self, predictions, labels):
        return tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy(predictions, labels))

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions), tf.float32)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
