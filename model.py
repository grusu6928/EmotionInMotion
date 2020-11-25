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
        # num channels,kernel size, stride size
        self.layer1 = tf.keras.layers.Conv2D(96, 11, strides = [4,4], activation = "relu")
        self.layer2 = tf.keras.layers.Conv2D(256, 5, strides = [1,1], activation = "relu")
        self.layer3 = tf.keras.layers.Conv2D(384, 3, strides = [1,1], activation = "relu")
        self.layer4 = tf.keras.layers.Conv2D(384, 3, strides = [1,1], activation = "relu")
        self.layer5 = tf.keras.layers.Conv2D(256, 3, strides = [1,1], activation = "relu")
        #“Maxpool-[kernelsize]-[stride size]”
        self.max1 = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2))
        self.max2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2)) 
        self.max3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2)) 
        # 259 is time series
        #self.u = tf.Variable(tf.random.truncated_normal((dims)))
        # 

    def FCN_layer(self, speech_spectrogram):
        #output of FCN encoder is a 3-dimensional
        # array of size F × T × C,where the F and T correspond to
        # the frequency and time domains of spectrogram and C is
        # channel size
        # lrn should take in 4d tensor as first input, but our conv2d uses 3d tensors, consider expand_dims or something else
        layer1out = tf.nn.lrn(self.layer1(speech_spectrogram))
        layer1out = self.max1(layer1out)
        layer2out = tf.nn.lrn(self.layer2(layer1out))
        layer2out = self.max2(layer2out)
        layer3out = self.layer3(layer2out)
        layer4out = self.layer4(layer3out)
        layer5out = self.layer5(layer4out)
        FCN_out = self.max3(layer5out)
        

        return FCN_out

    # def attention_layer(self, FCN_out):  # include softmax activation to determine classifications
    # self.dims = ()
    #     for i in range (3dim of FCN_out): 



    