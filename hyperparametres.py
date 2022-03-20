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


IMG_SIZE = 160

#loading data

X_train=load_data("X_train")
Y_train=load_data("Y_train")
X_valid=load_data("X_valid")
Y_valid=load_data("Y_valid")
X_test=load_data("X_test")
Y_test=load_data("Y_test")

#tuning our model

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(IMG_SIZE,IMG_SIZE,3)))
    
    #adding different layers
    model.add(layers.Conv2D(
        
        #tuning filters
        filters=hp.Int('conv_1_filter', min_value=32,max_value=128,step=16,default=64),

        #tuning kernel
        kernel_size = hp.Choice('conv_1_kernel',values = [2,3,5],default=2),

        input_shape = (IMG_SIZE,IMG_SIZE,3),

        #tuning activation mode
        activation = hp.Choice('conv_1_activation',values = ['relu','tanh'],default = 'relu')))

    #pooling
    model.add(layers.MaxPooling2D((2,2)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=hp.Float('dropout_1',min_value=0.0,max_value=0.5,default=0.25,step=0.05,)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32,max_value=128,step=16,default=96),
        kernel_size = hp.Choice('conv_2_kernel',values = [2,3,5],default=5),
        activation = hp.Choice('conv_2_activation',values = ['relu','tanh'],default = 'relu')))

    #adding other layers
    model.add(layers.MaxPooling2D((2,2)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=hp.Float('dropout_2',min_value=0.0,max_value=0.5,default=0.45,step=0.05,)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv_3_filter', min_value=32,max_value=128,step=16,default=64),
        kernel_size = hp.Choice('conv_3_kernel',values = [2,3,5],default=3),
        activation = hp.Choice('conv_3_activation',values = ['relu','tanh'],default = 'tanh')))


    model.add(layers.MaxPooling2D((2,2)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=hp.Float('dropout_3',min_value=0.0,max_value=0.5,default=0.4,step=0.05,)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv_4_filter', min_value=32,max_value=128,step=16,default=112),
        kernel_size = hp.Choice('conv_4_kernel',values = [2,3,5],default=5),
        activation = hp.Choice('conv_4_activation',values = ['relu','tanh'],default = 'relu')))


    model.add(layers.MaxPooling2D((2,2)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=hp.Float('dropout_4',min_value=0.0,max_value=0.5,default=0.15,step=0.05,)))

    #flatten to have a row
    model.add(layers.Flatten())


    model.add(layers.Dense(
        units=hp.Int('units',min_value=32,max_value=512,step=32,default=448),
        activation=hp.Choice('dense_activation',values=['relu','tanh'],default='relu')))

    #output of the cnn
    model.add(layers.Dense(1,activation='sigmoid'))

    #optimization and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-6),


            #choosing the loss function
            loss='binary_crossentropy',
            #choosing the metrics for mistakes
            metrics=['accuracy']
        )
    return model


""" hp.Float(
    'learning_rate',
    min_value=1e-4,
    max_value=1e-2,
    sampling='LOG',
    default=1e-3
    )"""

#specifying what kind of value we want to tune
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=50,
    executions_per_trial=2,
    overwrite=True,
    directory="O:\DermoTune",
    project_name="dermoscan_hyperparam_model",
)

#starting the tune of the model
tuner.search(X_train, Y_train, epochs=15,batch_size=45, validation_data=(X_valid, Y_valid))

#getting best models
models = tuner.get_best_models(num_models=2)
best_model = models[0]

#getting a summary of our model
best_model.summary()
tuner.results_summary()

#saving progression of the training by epoch
#callbacks = keras.callbacks.ModelCheckpoint(filepath=r'O:\DermoResult',save_freq='epoch')


#history = best_model.fit(x=X_train,y=Y_train,batch_size=50,epochs=20,validation_data=(X_valid,Y_valid),callbacks=callbacks)

#testing model
loss, acc = best_model.evaluate(x=X_test,y=Y_test)
print("loss: ",loss)
print("acc: ",acc)


