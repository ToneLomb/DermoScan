import pandas as pd
import numpy as np
import os
import cv2
import random
from  matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from save_load import *

#different directories where our dataset is saved

DATADIR = [r"D:\Scolaire\S4\Programmation\Projet\Dataset\train_sep",r"D:\Scolaire\S4\Programmation\Projet\Dataset\valid",r"D:\Scolaire\S4\Programmation\Projet\Dataset\test"]
#DATADIR = [r"/root/DermoScan/Projet/Dataset/archive/DerMel/train_sep",r"/root/DermoScan/Projet/Dataset/archive/DermMel/valid",r"/root/DermoScan/Projet/Dataset/archie/DerMel/test"]
CATEGORIES = ["Melanoma","NotMelanoma"]
IMG_SIZE = (150,150)


#creating different list that cointain our dataset splitted

training_data = []
validation_data = []
test_data = []

All_data = [training_data,validation_data,test_data]

#creating the future arrays we will work on

X_train = []
Y_train = []

X_valid = []
Y_valid = []

X_test = []
Y_test = []

couples = [[X_train,Y_train],[X_valid,Y_valid],[X_test,Y_test]]

#function that load all our images in our different directories

def create_data():
    for i in range(len(DATADIR)):
            for j in range(len(CATEGORIES)):
                category = CATEGORIES[j]
                path = os.path.join(DATADIR[i], category)
                for img in tqdm(os.listdir(path), desc="Chargement du dataset", colour="green"):
                    img_array = cv2.imread(os.path.join(path,img))
                    new_array = cv2.resize(img_array, IMG_SIZE)
                    All_data[i].append([new_array,j])


#randomizing all arrays for deep learning

def randomizing_data():
    for i in range(len(All_data)):
        random.shuffle(All_data[i])
        for features, labels in All_data[i]:
            couples[i][0].append(features)
            couples[i][1].append(labels)


#initialization

create_data()
randomizing_data()

#convert into numpy arrays

X_train = np.array(X_train).reshape(-1, IMG_SIZE ,3)
Y_train = np.array(Y_train).reshape(len(Y_train),1)

X_valid = np.array(X_valid).reshape(-1, IMG_SIZE,3)
Y_valid = np.array(Y_valid).reshape(len(Y_valid),1)

X_test = np.array(X_test).reshape(-1, IMG_SIZE,3)
Y_test = np.array(Y_test).reshape(len(Y_test),1)

#normalization of our data 0:255 --> 0:1

X_train = X_train.reshape(X_train.shape[0],IMG_SIZE,3) / 255
X_valid = X_valid.reshape(X_valid.shape[0],IMG_SIZE,3) / 255
X_test = X_test.reshape(X_test.shape[0],IMG_SIZE,3) / 255

#saving our dataset in order not to load it again

save_data(X_train,"X_train")
save_data(Y_train,"Y_train")
save_data(X_valid,"X_valid")
save_data(Y_valid,"Y_valid")
save_data(X_test,"X_test")
save_data(Y_test,"Y_test")

#different test

print("X_test shape :" , X_test.shape)
print("Y_test shape :" , Y_test.shape)