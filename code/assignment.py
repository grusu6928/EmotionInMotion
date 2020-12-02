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
    spec_train = []
    spec_test = []
    label_train = []
    label_test = []
    for sample in range(1,6):
        train = pd.read_csv("../data/train_sample"+str(sample)+".csv")
        test = pd.read_csv("../data/test_sample"+str(sample)+".csv")
        spec_test.append(test.iloc[:, 3:])
        label_test.append(test.iloc[:,1])

        spec_train.append(train.iloc[:, 3:])
    #print(spec_train.head(10))
        label_train.append(train.iloc[:,1])
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

def train(model, spec_train, label_train, shuffle=False, noprint=False):
    if(shuffle):
        indices = range(spec_train.shape[0])
        tf.random.shuffle(indices)
        spec_train = tf.gather(spec_train, indices)
        label_train = tf.gather(label_train, indices)
    loss_list = []
    for i in range(0,len(spec_train),model.batch_size):
        if model.batch_size + i <= len(spec_train):
            spec=spec_train[i:i+model.batch_size]
            label = label_train[i:i+model.batch_size]
            # Implement backprop
            with tf.GradientTape() as tape:
                predictions = model.call(spec)
                loss = model.loss(predictions, label)
                #print(loss)
                loss_list.append(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_acc = model.accuracy(predictions, label)
            if not noprint:
                print("Accuracy on training set after {} training steps: {}".format(i, train_acc))
    return loss_list

def test(model, spec_test, label_test):
    test_accuracy = []
    #used for weighted accuracy
    num_correct = np.zeros(4)
    total = np.zeros(4)
    for i in range(0,len(spec_test),model.batch_size):
        if model.batch_size + i <= len(spec_test):
            spec = spec_test[i:i+model.batch_size]
            label = label_test[i:i+model.batch_size]
            predictions = model.call(spec)
            test_accuracy.append(model.accuracy(predictions, label))
            temp_correct, temp_total = model.accuracy_weighted(predictions, label)
            num_correct = num_correct + temp_correct
            total = total + temp_total
    print(num_correct)
    print(total)
    return(np.mean(test_accuracy), np.nanmean(num_correct/total))

def main():
    if "VISUALIZE" in sys.argv:
        visualize = True
    else:
        visualize = False
        print("To visualize loss, call 'VISUALIZE' argument.")
    if "NOPRINT" in sys.argv:
        noprint = True
    else:
        noprint = False
        print("To run without printing, call 'NOPRINT' argument.")
    if "SHUFFLE" in sys.argv:
        shuffle = True
    else:
        shuffle = False
        print("To shuffle in train, call 'SHUFFLE' argument.")
    print()

    print("Starting preprocessing...")
    spec_train, spec_test, label_train, label_test = load_data()
    print("Data loaded!")
    for sample in range(5):
        label_train[sample], label_test[sample] = one_hot(label_train[sample], label_test[sample])
    print("One hots created!")
    for sample in range(5):
        spec_train[sample], spec_test[sample] = reshape_spectrogram(spec_train[sample], spec_test[sample])
    print("Reshaped spectogram")
    print("Finished preprocessing!")
    sample_accuracy = []
    sample_accuracy_weighted = []
    #run each sample
    for sample in range(5):
        print("Sample #", sample)
        model = Model()
        print("Starting training...")
        loss_list = []
        for _i_ in range(model.num_epochs):
            if not noprint:
                print()
                print("Epoch " + str(_i_ + 1) + ":")
                print()
            loss_list = loss_list + train(model, spec_train[sample], label_train[sample], shuffle=shuffle, noprint=noprint)
        if(visualize == True):
            print()
            visualize_loss(loss_list)
            print()
        print("Finished traning!")
        acc, weight_acc = test(model, spec_test[sample], label_test[sample])
        sample_accuracy.append(acc)
        sample_accuracy_weighted.append(weight_acc)
        print("Final unweighted accuracy for sample " + str(sample + 1) + " : " + str(acc))
        print("Final weighted accuracy for sample " + str(sample + 1) + " : " + str(weight_acc))
        print()
    print()

    print("Final unweighted accuracies:")
    for sample in range(5):
        print("Sample " + str(sample + 1) + " unweighted accuracy: " + str(sample_accuracy[sample]))
    print("Average unweighted accuracy: " + str(sum(sample_accuracy)/len(sample_accuracy)))
    print()
    print("Final weighted accuracies:")
    for sample in range(5):
        print("Sample " + str(sample + 1) + " weighted accuracy: " + str(sample_accuracy_weighted[sample]))
    print("Average weighted accuracy: " + str(sum(sample_accuracy_weighted)/len(sample_accuracy_weighted)))



if __name__ == '__main__':
    main()
