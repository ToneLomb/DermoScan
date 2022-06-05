import pickle
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import layers
from save_load import *
import os
from tensorflow.keras import regularizers
import cv2

model = keras.models.load_model('O:\DermoSave\V1')

model.save("O:\DermoSave\V2\model.h5")

DIR = r"O:\DermoData\modelTest"

normal = []
cut = []

path = os.path.join(DIR)

for img in (os.listdir(path)):
    img_array = cv2.imread(os.path.join(path,img))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    normal.append(img_array)
    new_array = cv2.resize(img_array,(160,160))
    cut.append(new_array)


test = np.array(cut).reshape(-1, 160,160 ,3)

print(test.shape)

prediction = model.predict(test)
print(prediction, "prediction uwu")


''' plt.figure(figsize=(12,8))
for i in range (len(normal)):
    plt.subplot(1,2,1)
    plt.imshow(normal[i])
    plt.subplot(1,2,2)
    plt.imshow(cut[i])
    plt.show()
*
'''


