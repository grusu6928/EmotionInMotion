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
    def __init__(self):
        super(Model, self).__init__()
        tf.keras.backend.set_floatx('float64')
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.batch_size = 60
        self.num_epochs = 200
        # num channels,kernel size, stride size
        self.layer1 = tf.keras.layers.Conv1D(5, 11, strides = 4, activation = "relu", input_shape = (259,1))
        self.layer2 = tf.keras.layers.Conv1D(10, 5, strides = 1, activation = "relu")
        self.layer3 = tf.keras.layers.Conv1D(20, 3, strides = 1, activation = "relu")
        self.layer4 = tf.keras.layers.Conv1D(30, 3, strides = 1, activation = "relu")
        #self.layer5 = tf.keras.layers.Conv1D(256, 3, strides = 1, activation = "relu") #!!!!!!!!!!
        #“Maxpool-[kernelsize]-[stride size]”
        self.max1 = tf.keras.layers.MaxPool1D(pool_size = 3, strides = 2)
        self.max2 = tf.keras.layers.MaxPool1D(pool_size = 3, strides = 2)
        self.max3 = tf.keras.layers.MaxPool1D(pool_size = 3, strides = 2)

        #attention stuff
        self.u = tf.Variable(tf.random.truncated_normal((self.batch_size, 4, 1)))
        self.linear_layer = tf.keras.layers.Dense((4), activation="tanh", dtype='float32') #might be different f and t
        #self.test_layer = tf.keras.layers.Dense((4), activation="tanh", dtype='float32')
        self.lammda = .8

    def FCN_layer(self, speech_spectrogram):
        #output of FCN encoder is a 3-dimensional
        # array of size F × T × C,where the F and T correspond to
        # the frequency and time domains of spectrogram and C is
        # channel size
        # lrn should take in 4d tensor as first input, but our conv2d uses 3d tensors, consider expand_dims or something else
        #TODO: lrn is commented out of now
        #print("speech_spectrogram shape")
        #print(speech_spectrogram.shape)
        layer1out = self.layer1(speech_spectrogram)
        #print("layer one output shape:")
        #print(layer1out.shape)
        layer1out = self.max1(layer1out)
        #print("max one output shape:")
        #print(layer1out.shape)
        layer2out = self.layer2(layer1out)
        #print("layer two output shape:")
        #print(layer2out.shape)
        layer2out = self.max2(layer2out)
        #print("max two output shape:")
        #print(layer2out.shape)
        layer3out = self.layer3(layer2out)
        #print("layer three output shape:")
        #print(layer3out.shape)
        layer4out = self.layer4(layer3out)
        #print("layer four output shape:")
        #print(layer4out.shape)
        #layer5out = self.layer5(layer4out)
        #print("layer five output shape:")
        #print(layer5out.shape)
        FCN_out = self.max3(layer4out)
        flattened_fcn_output = tf.keras.layers.Flatten()(FCN_out)

        #print("flattened fcn shape")
        #print(flattened_fcn_output.shape)
        linear_output = self.linear_layer(flattened_fcn_output)

        return linear_output

    def attention_layer(self, fcn_output):

        linear_output = tf.expand_dims(fcn_output, axis=2)
        e_tensor = tf.matmul(linear_output, tf.transpose(self.u,  perm=[0,2,1]))
        #print("e_tensor shape")
        #print(e_tensor.shape)
        exp_tensor = tf.math.exp(self.lammda * e_tensor)
        #print("exp_tensor shape")
        #print(exp_tensor.shape)
        alpha_tensor = exp_tensor / tf.reduce_sum(exp_tensor)
        #print("alpha_tensor shape")
        #print(alpha_tensor.shape)
        alpha_times_A = alpha_tensor * linear_output
        #print("alpha_times_A shape")
        #print(alpha_times_A.shape)
        c = tf.reduce_sum(alpha_times_A, axis = 2) #might need an axis
        #print("c shape")
        #print(c.shape)
        return c #this should be 24 by 4

    def call(self, spectogram):
        fcn_output = self.FCN_layer(spectogram)
        #print("fcn output shape:")
        #print(fcn_output.shape)
        #print("fcn finished!")
        attention_output = self.attention_layer(fcn_output)
        #self.test_layer(tf.flatten(fcn_output))
        #softmax
        return tf.nn.softmax(attention_output)

    def loss(self, predictions, labels):
        #print("predictions shape, labels shape")
        #print(predictions.shape)
        #print(labels.shape)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
        return tf.reduce_sum(loss)

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        #return tf.reduce_mean(tf.argmax(predictions, axis=0) == labels)
