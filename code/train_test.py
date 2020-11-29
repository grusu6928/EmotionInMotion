import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import os # interface with underlying OS that python is running on
import sys

df = pd.read_csv("data/full_data.csv")
df = df.sample(frac=1)
df.reset_index(inplace=True)
df.drop(columns=['Unnamed: 0','index'],inplace=True)
print(df.head(10))
print(df.shape)

### SAMPLE 1 ###
test1 = df.iloc[:102,:]
test1.reset_index(inplace=True)
test1.drop(columns=['index'],inplace=True)
print()
print("Test1")
print(test1.head(10))
print(test1.tail(10))
print(test1.shape)
test1.to_csv("data/test_sample1.csv")

train1 = df.iloc[102:,:]
train1.reset_index(inplace=True)
train1.drop(columns=['index'],inplace=True)
print()
print("Train1")
print(train1.head(10))
print(train1.tail(10))
print(train1.shape)
train1.to_csv("data/train_sample1.csv")

### SAMPLE 2 ###
test2 = df.iloc[102:204,:]
test2.reset_index(inplace=True)
test2.drop(columns=['index'],inplace=True)
print()
print("Test2")
print(test2.head(10))
print(test2.tail(10))
print(test2.shape)
test2.to_csv("data/test_sample2.csv")

train2_first = df.iloc[:102,:]
train2_second = df.iloc[204:,:]
train2 = pd.concat([train2_first,train2_second])
train2.reset_index(inplace=True)
train2.drop(columns=['index'],inplace=True)
print()
print("Train2")
print(train2.head(10))
print(train2.tail(10))
print(train2.shape)
train2.to_csv("data/train_sample2.csv")

### SAMPLE 3 ###
test3 = df.iloc[204:306,:]
test3.reset_index(inplace=True)
test3.drop(columns=['index'],inplace=True)
print()
print("Test3")
print(test3.head(10))
print(test3.tail(10))
print(test3.shape)
test3.to_csv("data/test_sample3.csv")

train3_first = df.iloc[:204,:]
train3_second = df.iloc[306:,:]
train3 = pd.concat([train3_first,train3_second])
train3.reset_index(inplace=True)
train3.drop(columns=['index'],inplace=True)
print()
print("Train3")
print(train3.head(10))
print(train3.tail(10))
print(train3.shape)
train3.to_csv("data/train_sample3.csv")

### SAMPLE 4 ###
test4 = df.iloc[306:408,:]
test4.reset_index(inplace=True)
test4.drop(columns=['index'],inplace=True)
print()
print("Test4")
print(test4.head(10))
print(test4.tail(10))
print(test4.shape)
test4.to_csv("data/test_sample4.csv")

train4_first = df.iloc[:306,:]
train4_second = df.iloc[408:,:]
train4 = pd.concat([train4_first,train4_second])
train4.reset_index(inplace=True)
train4.drop(columns=['index'],inplace=True)
print()
print("Train4")
print(train4.head(10))
print(train4.tail(10))
print(train4.shape)
train4.to_csv("data/train_sample4.csv")

### SAMPLE 5 ###
test5 = df.iloc[408:,:]
test5.reset_index(inplace=True)
test5.drop(columns=['index'],inplace=True)
print()
print("Test5")
print(test5.head(10))
print(test5.tail(10))
print(test5.shape)
test5.to_csv("data/test_sample5.csv")

train5 = df.iloc[:408,:]
train5.reset_index(inplace=True)
train5.drop(columns=['index'],inplace=True)
print()
print("Train5")
print(train5.head(10))
print(train5.tail(10))
print(train5.shape)
train5.to_csv("data/train_sample5.csv")