from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import pandas as pd 
import random as rd
import math

from model import Model

sample = 1

def load_data():
    train = pd.read_csv("data/train_sample"+str(sample)+".csv")
    test = pd.read_csv("data/test_sample"+str(sample)+".csv")
    spec_test = test.iloc[:, 3:]
    label_test = test.iloc[:,1]

    spec_train = train.iloc[:, 3:]
    print(spec_train.head(10))
    label_train = train.iloc[:,1]
    input()
    print(label_train.head(10))
    pass

def one_hot(label_train, label_test):
    # CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
    lb = LabelEncoder()

    label_train = to_categorical(lb.fit_transform(label_train))
    label_test = to_categorical(lb.fit_transform(label_test))

    return label_train, label_test

def reshape_spectrogram(spec_train, spec_test):
    spec_train = spec_train[:,:,np.newaxis]
    spec_test = spec_test[:,:,np.newaxis]

    return spec_train,spec_test

    

def main():
    load_data()


if __name__ == '__main__':
    main()

