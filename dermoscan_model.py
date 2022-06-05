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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

IMG_SIZE=160

X_train=load_data("X_train")
Y_train=load_data("Y_train")
X_valid=load_data("X_valid")
Y_valid=load_data("Y_valid")
X_test=load_data("X_test")
Y_test=load_data("Y_test")


"""def create_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(150, 150, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))
    model.summary()
    return model"""




def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(80, (2, 2) , input_shape = (IMG_SIZE, IMG_SIZE, 3),activation = 'tanh', activity_regularizer=regularizers.l1(1e-4)), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Dropout(0.15), 
    tf.keras.layers.Conv2D(48, (5, 5) , activation = 'tanh', activity_regularizer=regularizers.l1(1e-4)), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Dropout(0.05), 
    tf.keras.layers.Conv2D(112, (2, 2) , activation = 'tanh', activity_regularizer=regularizers.l1(1e-4)), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Dropout(0.45), 
    tf.keras.layers.Conv2D(112, (3, 3), activation = 'relu', activity_regularizer=regularizers.l1(1e-4)), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Dropout(0.4), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation = 'relu', activity_regularizer=regularizers.l1(1e-4)), 
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
    return model

model = create_model()

callbacks = keras.callbacks.ModelCheckpoint(filepath=r'O:\DermoResult', save_freq='epoch')

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics = ['acc'])

history = model.fit(x=X_train, y=Y_train, batch_size=60, epochs=75, validation_data=(X_valid, Y_valid), callbacks=callbacks)

loss, acc = model.evaluate(x=X_test, y=Y_test)

model.save(r'O:\DermoSave')

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history["acc"], label="train")
plt.plot(history.history["val_acc"], label="val")
plt.title("train vs val accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("train vs val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()


print("acc test :", acc)
print("loss test :", loss)


