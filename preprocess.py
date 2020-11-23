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

path = "../Audio_Speech_Actors_01-24"
emotion = []
file_path = []
actor = []
for i in range(1,25):
    num = str(i)
    if len(num) == 1:
        num = "0"+num
    audio = path + "/Actor_" + num
    filename = os.listdir(audio)
    for f in filename:
        split_name = f.split('.')[0].split('-')
        emotion.append(int(split_name[2]))
        actor.append(int(split_name[6]))
        file_path.append(audio + '/' + f)
audio_df = pd.DataFrame(emotion)
audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
audio_df = pd.concat([audio_df,pd.DataFrame(actor)],axis=1)
audio_df.columns = ['emotion','actor']
audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['file_path'])],axis=1)
audio_df.set_index('emotion',inplace=True)
audio_df.drop(index=['calm','fear','disgust','surprise'],inplace=True)
audio_df.reset_index(inplace=True)

df = pd.DataFrame(columns=['mel_spectrogram'])
counter=0
for index,path in enumerate(audio_df.file_path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
    db_spec = librosa.power_to_db(spectrogram)
    log_spectrogram = np.mean(db_spec, axis = 0)
    df.loc[counter] = [log_spectrogram]
    #print(df.head())
    counter=counter+1
df = df.mel_spectrogram.apply(pd.Series)
audio_df = pd.concat([audio_df,df],axis=1)
audio_df.drop(columns='file_path',inplace=True)
print(audio_df.head(10))
print(audio_df.shape)
audio_df.dropna(inplace=True)
print()
print("DROPPED")
print(audio_df.head(10))
print(audio_df.shape)