import tensorflow as tf
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import layers
from save_load import *
import keras_tuner as kt

IMG_SIZE = 150

X_train=load_data("X_train")
Y_train=load_data("Y_train")
X_valid=load_data("X_valid")
Y_valid=load_data("Y_valid")
X_test=load_data("X_test")
Y_test=load_data("Y_test")

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(150,150,3)))
    model.add(layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32,max_value=128,step=16,default=32),
        kernel_size = hp.Choice('conv_1_kernel',values = [2,3,5],default=3),
        input_shape = (IMG_SIZE,IMG_SIZE,3),
        activation = hp.Choice('conv_1_activation',values = ['relu','sigmoid','tanh'],default = 'relu')))

    model.add(layers.MaxPooling2D((2,2)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=hp.Float('dropout_1',min_value=0.0,max_value=0.5,default=0.25,step=0.05,)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32,max_value=128,step=16,default=64),
        kernel_size = hp.Choice('conv_2_kernel',values = [2,3,5],default=3),
        activation = hp.Choice('conv_2_activation',values = ['relu','sigmoid','tanh'],default = 'relu')))

    model.add(layers.MaxPooling2D((2,2)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=hp.Float('dropout_2',min_value=0.0,max_value=0.5,default=0.25,step=0.05,)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv_3_filter', min_value=32,max_value=128,step=16,default=128),
        kernel_size = hp.Choice('conv_3_kernel',values = [2,3,5],default=3),
        activation = hp.Choice('conv_3_activation',values = ['relu','sigmoid','tanh'],default = 'relu')))


    model.add(layers.MaxPooling2D((2,2)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=hp.Float('dropout_3',min_value=0.0,max_value=0.5,default=0.25,step=0.05,)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv_4_filter', min_value=32,max_value=128,step=16,default=128),
        kernel_size = hp.Choice('conv_4_kernel',values = [2,3,5],default=3),
        activation = hp.Choice('conv_4_activation',values = ['relu','sigmoid','tanh'],default = 'relu')))


    model.add(layers.MaxPooling2D((2,2)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=hp.Float('dropout_4',min_value=0.0,max_value=0.5,default=0.25,step=0.05,))) 


    model.add(layers.Flatten())


    model.add(layers.Dense(
        inputs=hp.Int('units',min_value=32,max_value=512,step=32,default=128),
        activation=hp.Choice('dense_activation',values=['relu','sigmoid','tanh'])))


    model.add(layers.Dense(1,activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float(
                'learning_rate',
                min_value=1e-4,
                max_value=1e-2,
                sampling='LOG',
                default=1e-3
                )
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    return model

model = build_model(kt.HyperParameters())

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="chemin_du_pc_victor",
    project_name="dermoscan_hyperparam_model",
)

tuner.search(X_train, Y_train, epochs=5, validation_data=(X_valid, Y_valid))
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()
tuner.results_summary()

history = model.fit(x=X_train,y=Y_train,batch_size=50,epochs=20,validation_data=(X_valid,Y_valid),callbacks=callbacks)

loss, acc = model.evaluate(x=X_test,y=Y_test)


plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("train vs val accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.subplot(2,1,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("train vs val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()