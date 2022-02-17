import pandas as pd
import numpy as np
import os
import cv2
import random
from  matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

#different directories where our dataset is saved

DATADIR = [r"D:\Scolaire\S4\Programmation\Projet\Dataset\train_sep",r"D:\Scolaire\S4\Programmation\Projet\Dataset\valid",r"D:\Scolaire\S4\Programmation\Projet\Dataset\test"]
#DATADIR = [r"/root/DermoScan/Projet/Dataset/archive/DerMel/train_sep",r"/root/DermoScan/Projet/Dataset/archive/DermMel/valid",r"/root/DermoScan/Projet/Dataset/archie/DerMel/test"]
CATEGORIES = ["Melanoma","NotMelanoma"]


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
                    new_array = cv2.resize(img_array, (150,150))
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

X_train = np.array(X_train).reshape(-1, 150,150,3)
Y_train = np.array(Y_train).reshape(len(Y_train),1)

X_valid = np.array(X_valid).reshape(-1, 150,150,3)
Y_valid = np.array(Y_valid).reshape(len(Y_valid),1)

X_test = np.array(X_test).reshape(-1, 150,150,3)
Y_test = np.array(Y_test).reshape(len(Y_test),1)

#saving our dataset in order not to load it again

pickle_out = open("X_train","wb")
pickle.dump(X_train,pickle_out)
pickle_out.close()

pickle_out = open("Y_train","wb")
pickle.dump(Y_train,pickle_out)
pickle_out.close()

pickle_out = open("X_valid","wb")
pickle.dump(X_valid,pickle_out)
pickle_out.close()

pickle_out = open("Y_valid","wb")
pickle.dump(Y_valid,pickle_out)
pickle_out.close()

pickle_out = open("X_test","wb")
pickle.dump(X_test,pickle_out)
pickle_out.close()

pickle_out = open("Y_test","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()

#different test

print("X_test shape :" , X_test.shape)
print("Y_test shape :" , Y_test.shape)