from __future__ import print_function

from datetime import datetime
import pandas as pd
import numpy as np
import io
import itertools
import pickle
import shutil
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import h5py

from livelossplot.keras import PlotLossesCallback
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.layers import Dense, Reshape, Dropout, Input, LSTM, Bidirectional
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras import callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint, Callback
from tensorflow.keras import backend as k

#custome libaries
from data_preprocessing_CICDS import cicids_data
from data_preprocessing_CICDS import balance_data
from Autoencoder_CICDS_model import build_AE

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


logdir = "/home/vibek/Anomanly_detection_packages/DNN_Package/logs_directory/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)

params = {'dataset': 'CICIDS-2017'}

print("Loading dataset CICIDS 2017 .....\n")
train_data, test_data, train_labels,  test_labels = cicids_data(params)
#print("train shape: ", train_data.shape)
#print("test shape: ", test_data.shape)
#print("train_label shape: ", train_labels.shape)
#print("test_label shape: ", test_labels.shape)

print("Loading AutoEncoder model....\n")
autoencoder_1, encoder_1, autoencoder_2, encoder_2, autoencoder_3, encoder_3, sSAE, SAE_encoder = build_AE()
print("value of SAE_encoder:\n", SAE_encoder.output)
print("value of SAE_encoder:\n", SAE_encoder.input)

#TimesereisGenerator
time_steps = 1
batch_size = 1024
epochs = 1000

# updatable plot
class Plot_loss_accuracy(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
        
plot = Plot_loss_accuracy()

print('Finding feature importances.....')

def find_importances(X_train, Y_train):
    model = ExtraTreesClassifier()
    model = model.fit(X_train, Y_train)
    
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]  # Top ranking features' indices
    return importances, indices, std


# Plot the feature importances of the forest
def plot_feature_importances(X_train, importances, indices, std, title):
   #tagy
#     for f in range(X_train.shape[1]):
#         print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    plt.figure(num=None, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title(title)
    width=5
    plt.bar(range(X_train.shape[1]), importances[indices],
          width=5, color="r", yerr=std[indices], align="center") #tagy 1.5 > .8
    plt.xticks(range(X_train.shape[1]), indices)
    #plt.axis('tight')
    plt.xlim([-1, X_train.shape[1]]) # -1 tagy
    plt.show()

# Neural network is classified with correct 'attack or not' labels
X_nn_train = np.concatenate((train_labels, train_data), axis=1)
nn_importances, nn_indices, nn_std = find_importances(X_nn_train,
                                                      train_labels)
plot_feature_importances(X_nn_train,
                        nn_importances, nn_indices, nn_std, title='Feature importances (CICIDS-2017)')

def build_model_LSTM():
# 1. define the network
    mlp0 = Dense(units=32, activation='relu')(SAE_encoder.output)

    lstm_reshape = Reshape((1, 32))(mlp0)

    lstm1 = LSTM(units=24, activation='tanh', return_sequences=True)(lstm_reshape)

    lstm1_reshape = Reshape((1, 24))(lstm1)

    lstm2 = LSTM(units=16, activation='tanh', return_sequences=True)(lstm1_reshape)

    lstm2_reshape = Reshape((1, 16))(lstm2)

    lstm3 = LSTM(units=10, activation='tanh', return_sequences=True)(lstm2_reshape)

    lstm3_reshape = Reshape((1, 10))(lstm3)

    lstm4 = LSTM(units=6, activation='tanh', return_sequences=True)(lstm3_reshape)

    lstm4_reshape = Reshape((1, 6))(lstm4)

    lstm5 = LSTM(units=2, activation='sigmoid')(lstm4_reshape)

    model = Model(SAE_encoder.input, lstm5)
    model.summary()
    plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/model_details_cicds.png',show_shapes=True)

    return model
"""
# try using different optimizers and different optimizer configs
    start = time.time()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    print ("Compilation Time:", time.time() - start)
    plot_losses = PlotLossesCallback()

#save model and the values
    save_model = ModelCheckpoint(filepath="/home/vibek/Anomanly_detection_packages/DNN_Package/checkpoint-{epoch:02d}.hdf5", verbose=1, monitor='val_acc', save_best_only=True)	
    csv_logger = CSVLogger('/home/vibek/Anomanly_detection_packages/DNN_Package/training_set_dnnanalysis_cicds.csv', separator=',', append=False)
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
    callbacks = [save_model, csv_logger, tensorboard_callback, early_stopping_monitor]

    global_start_time = time.time()
    print("Start Training...")
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels), callbacks=callbacks)
    model.save("/home/vibek/Anomanly_detection_packages/DNN_Package/best_model_cicds.hdf5")
    print("Done Training...")
"""
    