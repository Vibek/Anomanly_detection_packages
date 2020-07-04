import os
import numpy as np

from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Reshape, Dropout

#custome libaries
from data_preprocessing_unsw import unsw_data_common
from Autoencoder_unsw_model import build_unsw_AE

params = {'dataset': 'unsw_NB'}

#loading the data from data preprocessing
print("Loading.....")
train_data, train_labels, test_data, test_labels = unsw_data_common(params)
print("train shape: ", train_data.shape)
print("test shape: ", test_data.shape)
encoded_train_label = train_labels.reshape((-1, 1))
encoded_test_label = test_labels.reshape((-1, 1))
print("train_label shape: ", train_labels.shape)
print("test_label shape: ", test_labels.shape)

# build model
print("Build AE model")
autoencoder_1, encoder_1, autoencoder_2, encoder_2, autoencoder_3, encoder_3, sSAE, SAE_encoder = build_unsw_AE(rho=0.04)

autoencoder_1.summary()
encoder_1.summary()
autoencoder_2.summary()
encoder_2.summary()
autoencoder_3.summary()
encoder_3
sSAE.summary()
SAE_encoder.summary()

print("Start pre-training....")

# fit the first layer
print("First layer training....")

ae_1_filepath="/home/vibek/Anomanly_detection_packages/DNN_Package/Model_AE/best_ae_1.hdf5"
ae_1_point = ModelCheckpoint(filepath=ae_1_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_1_stops = EarlyStopping(monitor='val_loss', patience=10, mode='min')
autoencoder_1.fit(train_data, train_data, epochs=1000, batch_size=512, validation_data=(train_labels, train_labels), verbose=0, shuffle=True, callbacks=[ae_1_point, ae_1_stops])

autoencoder_1.load_weights('/home/vibek/Anomanly_detection_packages/DNN_Package/Model_AE/best_ae_1.hdf5')
first_layer_output = encoder_1.predict(train_data)  
test_first_out = encoder_1.predict(train_labels)
print("The shape of first layer output is: ", first_layer_output.shape)

# fit the second layer
print("Second layer training....")

ae_2_filepath="/home/vibek/Anomanly_detection_packages/DNN_Package/Model_AE/best_ae_2.hdf5"
ae_2_point = ModelCheckpoint(filepath=ae_2_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_2_stops = EarlyStopping(monitor='val_loss', patience=10, mode='min')
autoencoder_2.fit(first_layer_output, first_layer_output, epochs=1000, batch_size=512, verbose=0, validation_data=(test_first_out, test_first_out), shuffle=True, callbacks=[ae_2_point, ae_2_stops])

autoencoder_2.load_weights('/home/vibek/Anomanly_detection_packages/DNN_Package/Model_AE/best_ae_2.hdf5')
second_layer_output = encoder_2.predict(first_layer_output)
test_second_out = encoder_2.predict(test_first_out)
print("The shape of second layer output is: ", second_layer_output.shape)

# fit the third layer
print("Third layer training....")

ae_3_filepath="/home/vibek/Anomanly_detection_packages/DNN_Package/Model_AE/best_ae_3.hdf5"
ae_3_point = ModelCheckpoint(filepath=ae_3_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_3_stops = EarlyStopping(monitor='val_loss', patience=10, mode='min')
autoencoder_3.fit(second_layer_output, second_layer_output, epochs=1000, batch_size=512, verbose=0, validation_data=(test_second_out, test_second_out), shuffle=True, callbacks=[ae_3_point, ae_3_stops])
autoencoder_3.load_weights('/home/vibek/Anomanly_detection_packages/DNN_Package/Model_AE/best_ae_3.hdf5')

print("Pass the weights to SAE_encoder...")
SAE_encoder.layers[1].set_weights(autoencoder_1.layers[1].get_weights())  # first Dense
SAE_encoder.layers[2].set_weights(autoencoder_1.layers[2].get_weights())  # first BN
SAE_encoder.layers[3].set_weights(autoencoder_2.layers[1].get_weights())  # second Dense
SAE_encoder.layers[4].set_weights(autoencoder_2.layers[2].get_weights())  # second BN
SAE_encoder.layers[5].set_weights(autoencoder_3.layers[1].get_weights())  # third Dense
SAE_encoder.layers[6].set_weights(autoencoder_3.layers[2].get_weights())  # third BN

encoded_train_data = SAE_encoder.predict(train_data)
encoded_test_data = SAE_encoder.predict(train_labels)

print('encoded_train_data:', encoded_train_data)
print('encoded_test_data:', encoded_test_data)

print('SAE_encoder input:', SAE_encoder.input)
print('SAE_encoder output:', SAE_encoder.output)

np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/Encoded_data/encoded_train_unsw.npy', encoded_train_data)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/Encoded_data/train_label_unsw.npy', encoded_train_label)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/Encoded_data/encoded_test_unsw.npy', encoded_test_data)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/Encoded_data/test_label_unsw.npy', encoded_test_label)

print('encoded_test_label:', encoded_test_label)
print('encoded_train_label:', encoded_train_label)