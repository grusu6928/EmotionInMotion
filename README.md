## README

### Emotion in Motion - Attention Based Speech Emotion Classification
Deep Learning 2020 Final Project

#### Table of contents
* [Introduction](#introduction)
* [Dataset](#dataset)
* [Program Structure](#program-structure)
* [Necessary Packages](#necessary-packages)
* [Running the Model](#running-the-model)

#### Introduction 
The problem we are addressing is the difficulty of classifying human emotion, given variable length speech. Speech emotion recognition systems struggle with classifying emotion due to the abstract nature of emotion and the fact that human emotion can only be detected in small parts during long segments of speech. 

#### Dataset
[RAVDESS](https://smartlaboratory.org/ravdess/) (Ryerson Audio-Visual Database of Emotional Speech and Song)

#### Program Structure
The **`code`** directory is where the program resides. 
The **`data`** directory is created and populated with the preprocessed data after running `preprocess.py` and `train_test.py`

`model.py` - contains the FCN and attention layer, as well as the call function which does the forward and backward passes 
`preprocess.py` - aggregates raw data from the RAVDESS dataset performs initial feature extraction of mel-spectograms
`train_test.py` - splits preprocessed data into 5 unique train-test splits for five-fold cross validation mechanism
`assignment.py` - handles model creation, training and batching, and testing 

#### Necessary Packages
In addition to the standard CSCI 1470 virtual environment, the following packages are required to run the model: 
```
pip install librosa 
pip install pandas
```
We also include our own virtual environment with our own `requirements.txt`, which can be created by running 
`./create_venv`

#### Running the Model
After cloning the repo, creating and activating the virtual environment, 
and running `preprocess.py` and `train_test.py`, you can run the model and train via 
`./assignment.py`
`assignment.py` takes in any of the following optional flags:
`./assignment.py NOPRINT` is used to suppress printing the test and train accuracies for every sample and epoch 
`./assingment.py VISUALIZE` is used to plot and display the loss graphs for each sample 
`./assignment SHUFFLE` is used to randomly shuffle the images in the train set for each batch and epoch

