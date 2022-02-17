import pickle
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import layers

pickle_in = open("X_train_reshape","rb")
X_train = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("X_valid_reshape","rb")
X_valid = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("X_test_reshape","rb")
X_test = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("Y_train","rb")
Y_train = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("Y_valid","rb")
Y_valid = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("Y_test","rb")
Y_test = pickle.load(pickle_in)
pickle_in.close()


"""def create_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(150,150,3)))
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1,activation='softmax'))
    model.summary()
    return model"""



def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3) ,input_shape = (150,150,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3) ,activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3) ,activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation = 'relu'),
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])
    return model

model = create_model()

callbacks = keras.callbacks.ModelCheckpoint(filepath=r'D:\Scolaire\S4\Programmation\Projet\model',save_freq='epoch')

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),loss='binary_crossentropy',metrics = ['acc'])

history = model.fit(x=X_train,y=Y_train,batch_size=50,epochs=5,validation_data=(X_valid,Y_valid),callbacks=callbacks)

loss, acc = model.evaluate(x=X_valid,y=Y_valid)

print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

