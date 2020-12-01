from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import random as rd
import math
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import sys

from model import Model

sample = 1

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def load_data():
    train = pd.read_csv("../data/train_sample"+str(sample)+".csv")
    test = pd.read_csv("../data/test_sample"+str(sample)+".csv")
    spec_test = test.iloc[:, 3:]
    label_test = test.iloc[:,1]

    spec_train = train.iloc[:, 3:]
    #print(spec_train.head(10))
    label_train = train.iloc[:,1]
    #print(label_train.head(10))
    return spec_train, spec_test, label_train, label_test

def one_hot(label_train, label_test):
    # CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
    lb = LabelEncoder()

    label_train = to_categorical(lb.fit_transform(label_train))
    label_test = to_categorical(lb.fit_transform(label_test))

    return label_train, label_test

def reshape_spectrogram(spec_train, spec_test):
    spec_train = tf.expand_dims(spec_train, axis=2)
    spec_test = tf.expand_dims(spec_test, axis=2)
    return spec_train,spec_test

def train(model, spec_train, label_train):
    #indices = range(spec_train.shape[0])
    #indices = tf.random.shuffle(indices)
    #spec_train = tf.gather(spec_train, indices)
    #label_train = tf.gather(label_train, indices)
    loss_list = []
    for i in range(0,len(spec_train),model.batch_size):
        if model.batch_size + i <= len(spec_train):
            spec=spec_train[i:i+model.batch_size]
            label = label_train[i:i+model.batch_size]
            # Implement backprop: 
            with tf.GradientTape() as tape:
                predictions = model.call(spec)
                loss = model.loss(predictions, label)
                #print(loss)
                loss_list.append(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_acc = model.accuracy(predictions, label)
            print("Accuracy on training set after {} training steps: {}".format(i, train_acc))
    return loss_list

def test(model, spec_test, label_test):
    test_accuracy = []
    for i in range(0,len(spec_test),model.batch_size):
        if model.batch_size + i <= len(spec_test):
            spec = spec_test[i:i+model.batch_size]
            label = label_test[i:i+model.batch_size]
            # Implement backprop: 
            predictions = model.call(spec) # this calls the call function conveniently 
            test_accuracy.append(model.accuracy(predictions, label))
    return(np.mean(test_accuracy))

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "VISUALIZE":
        visualize = True
    else:
        visualize = False
        print("To visualize loss, call 'python assignment.py VISUALIZE'")
        print()
    print("Starting preprocessing...")
    spec_train, spec_test, label_train, label_test = load_data()
    print("Data loaded!")
    label_train, label_test = one_hot(label_train, label_test)
    print("One hot created!")
    spec_train, spec_test = reshape_spectrogram(spec_train, spec_test)
    print("Reshaped spectogram")
    print("Finished preprocessing!")
    f = spec_train.shape[0] #frequency
    t = spec_train.shape[1] #time
    model = Model()
    print("Starting training...")
    loss_list = []
    for _i_ in range(model.num_epochs):
        print()
        print("Epoch " + str(_i_ + 1) + ":")
        print()
        loss_list = loss_list + train(model, spec_train, label_train)
    if(visualize == True):
        print()
        visualize_loss(loss_list)
        print()
    print("Finished training!")
    print("Final accuracy: " + str(test(model, spec_test, label_test)))


if __name__ == '__main__':
    main()
