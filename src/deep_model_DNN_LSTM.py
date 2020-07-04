from __future__ import print_function

#python libararies
from datetime import datetime
import pandas as pd
import numpy as np
import io
import itertools
import pickle
import shutil
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot
from IPython.display import clear_output

#sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
import h5py
from pathlib import Path

#import tensorflow
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from livelossplot.keras import PlotLossesCallback
from keras.models import Model
from keras.utils import plot_model
from keras.layers import concatenate, ZeroPadding2D
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.layers import Dense, Reshape, Dropout, Input, LSTM, Bidirectional, Flatten

#from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing import sequence
from keras.models import model_from_json
from tensorflow.keras.datasets import imdb
from tensorflow.keras import callbacks
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint, Callback
from tensorflow.keras import backend as k

#custome libaries
from data_preprocessing_IoT import IoT_data
#from data_preprocessing_unsw import unsw_data
#from data_preprocessing_CICDS import cicids_data
#from data_preprocessing_CICDS import balance_data


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

params = {'dataset': 'IoT_data'}





#TimesereisGenerator
time_steps = 1
batch_size = 512
epochs = 100

print("Loading dataset IoT-23 .....\n")
train_data, test_data, train_labels, test_labels = IoT_data(params)

#DNN model load
print('Loading DNN model......\n')
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
	model_path = "/home/vibek/Anomanly_detection_packages/DNN_Package/Model_DNN/"
	model_name = model_path+"best_model.hdf5"
	model_DNN = load_model(model_name, compile=False)
	model_DNN.summary()
	
#LSTM model load
print('Loading LSTM model......\n')
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
	model_path = "/home/vibek/Anomanly_detection_packages/DNN_Package/Model_LSTM/"
	model_name = model_path+"best_model_cicds.hdf5"
	model_LSTM = load_model(model_name, compile=False)
	model_LSTM.summary()

#Use the data array same szie to feed the model
#random_indices = np.random.choice(train_data_unsw.shape[0], train_data_cicds.shape[0], replace=True) 
#new_train_data_unsw = train_data_unsw[random_indices]
#print('New shape of UNSW data:', new_train_data_unsw.shape)

#random_indices_labels = np.random.choice(train_labels_unsw.shape[0], train_labels_cicds.shape[0], replace=True) 
#new_train_labels_unsw = train_labels_unsw[random_indices_labels]
#print("New shape of UNSW labels:", new_train_labels_unsw.shape)	


#new input shape
new_input_shape = (None, 13)
inputTensor = Input(shape=(13, ))
model_DNN._layers[0].batch_input_shape = new_input_shape
model_LSTM._layers[0].batch_input_shape = new_input_shape

new_model_DNN = model_from_json(model_DNN.to_json())
new_model_LSTM = model_from_json(model_LSTM.to_json())

new_model_DNN.name = 'model_DNN'
new_model_LSTM.name = 'model_LSTM'

print('The size of input shape:', new_input_shape)

new_model_DNN.layers[0].set_weights(model_DNN.layers[0].get_weights())
new_model_LSTM.layers[0].set_weights(model_LSTM.layers[0].get_weights())

# try using different optimizers and different optimizer configs   
start = time.time() 
new_model_DNN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
new_model_DNN.summary()   

start = time.time() 
new_model_LSTM.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
new_model_LSTM.summary()  

models_output = [new_model_DNN.output, new_model_LSTM.output]
models_input = [new_model_DNN.input, new_model_LSTM.input]
models = [new_model_DNN, new_model_LSTM]
outputTensor = [m(inputTensor) for m in models]

# now change the name of the layer inplace.
new_model_DNN.get_layer(name='dropout_1').name='dropout_11'
new_model_DNN.get_layer(name='dropout_2').name='dropout_21'
new_model_DNN.get_layer(name='dropout_3').name='dropout_31'
new_model_LSTM.get_layer(name='input_4').name='input_41'
new_model_LSTM.get_layer(name='dense_7').name='dense_71'
new_model_LSTM.get_layer(name='batch_normalization_4').name='batch_normalization_41'
new_model_LSTM.get_layer(name='dense_8').name='dense_81'
new_model_LSTM.get_layer(name='batch_normalization_5').name='batch_normalization_51'
new_model_LSTM.get_layer(name='dense_9').name='dense_91'
new_model_LSTM.get_layer(name='batch_normalization_6').name='batch_normalization_61'
new_model_LSTM.get_layer(name='dense_13').name='dense_131'

#Merged layers of two models
temp_model= concatenate(outputTensor, axis=-1)
x = BatchNormalization()(temp_model)
mlp0 = Dense(30, activation='relu')(x)
mlp0_drop = Dropout(0.3)(mlp0)
x1 = BatchNormalization()(mlp0_drop)
mlp1 = Dense(16, activation='relu')(x1)
mlp1_drop = Dropout(0.3)(mlp1)
x2 = BatchNormalization()(mlp1_drop)
mlp2 = Dense(9, activation='relu')(x2)
mlp2_drop = Dropout(0.3)(mlp2)
#mlp3 = Dense(2, activation='sigmoid')(mlp1_drop)
mlp4 = Dense(2, activation='sigmoid')(mlp2_drop)

final_model= Model(inputTensor, mlp4, name="new_model")
#final_model.get_layer(name='input_4').name='input_411'
plot_model(final_model,to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/model_details_final.png', show_shapes=True)

# try using different optimizers and different optimizer configs
start = time.time()
final_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print ("Compilation Time:", time.time() - start)
final_model.summary()

#
"""
#new input shape
new_input_shape = (None, 13)
inputTensor = Input(shape=(13, ))
model_LSTM._layers[0].batch_input_shape = new_input_shape

new_model_LSTM = model_from_json(model_LSTM.to_json())
new_model_LSTM.layers[0].set_weights(model_LSTM.layers[0].get_weights())
new_model_LSTM.name = 'update_model_LSTM'
new_model_LSTM.summary()

new_model_LSTM.layers.pop()
new_model_LSTM.outputs = [new_model_LSTM.layers[-1].output]
new_model_LSTM.layers[-1].outbound_nodes = []
new_model_LSTM.summary()

mlp0 = Dense(2, activation='relu')(new_model_LSTM.output)
mlp0_drop = Dropout(0.3)(mlp0)
x1 = BatchNormalization()(mlp0_drop)
Fc_layer = Dense(2, activation='sigmoid')(x1)
new_model =  Model(new_model_LSTM.input, Fc_layer)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process 
i=0
for layer in new_model_LSTM.layers:
    layer.trainable = True
    i = i+1
    print(i,layer.name)

    # loop over the layers in the model and show which ones are trainable
# or not
for layer in new_model_LSTM.layers:
	print("{}: {}".format(layer, layer.trainable))

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
# try using different optimizers and different optimizer configs   
start = time.time() 
new_model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
new_model.summary()

# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random

#save model and the values
save_model = ModelCheckpoint(filepath="/home/vibek/Anomanly_detection_packages/DNN_Package/checkpoint-{epoch:02d}.hdf5", verbose=1, monitor='val_acc', save_best_only=True)	
csv_logger = CSVLogger('/home/vibek/Anomanly_detection_packages/DNN_Package/training_set_dnnanalysis_IoT_esemble.csv', separator=',', append=False)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
callbacks = [save_model, csv_logger, tensorboard_callback, early_stopping_monitor]

global_start_time = time.time()
print("Start Training...")
new_model.fit(train_data, test_data, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_data=(train_labels, test_labels), callbacks=callbacks)
new_model.save("/home/vibek/Anomanly_detection_packages/DNN_Package/best_model_IoT.hdf5")
print("Done Training...")

predictions_valid = new_model.predict(train_labels, batch_size=batch_size, verbose=1)
score = log_loss(test_labels, predictions_valid)
print('prediction_score:', predictions_valid)
print('Log_score:', score)

	sclf = StackingClassifier(classifiers=m, 
                          meta_classifier=lr)
	clf_list = [m, sclf]
	print('list of model:', clf_list)
	label = ['DNN_Model', 'LSTM Model', 'Stacking Classifier']

	scores = cross_val_score(clf_list, train_data, test_data, cv=3, scoring='accuracy')
	print('Stacked accuracy_score:', sclf)
	print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

"""
