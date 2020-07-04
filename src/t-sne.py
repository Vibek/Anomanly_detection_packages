from __future__ import print_function

from datetime import datetime
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from matplotlib import pyplot
import io
import itertools
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.manifold import TSNE
from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
from yellowbrick.text import TSNEVisualizer

#custome libaries
from utils import *
from data_preprocessing_NetML import *
from data_preprocessing_LITNET import LITNET_data
from data_preprocessing_IoT import IoT_data_common
from data_preprocessing_IoT import balance_data
from Autoencoder_IoT_model import build_iot_AE
from Autoencoder_NetML_model import build_NetML_AE
from Autoencoder_LITNET_model import build_LITNET_AE
'''
params = {'dataset': 'IoT-23'}

###LITNET-2020 dataset####
print("Loading dataset LITNET-2020.....\n")
train_data, train_labels, test_data, test_labels = LITNET_data(params)
print("LITNET_data train shape: ", train_data.shape)
print("LITNET_data train_label shape: ", test_data.shape)
print("LITNET_data validation shape: ", train_labels.shape)
print("LITNET_data Validation_label shape: ", test_labels.shape)
test_label_original = np.argmax(test_labels, axis=1)
train_label_original = np.argmax(test_data, axis=1)


###IoT-23 dataset####
print("Loading dataset IoT-23.....\n")
train_data, train_labels, test_data, test_labels = IoT_data_common(params)
print("train shape: ", train_data.shape)
print("test shape: ", test_data.shape)
print("train_label shape: ", train_labels.shape)
print("test_label shape: ", test_labels.shape)
#test_label_original = np.argmax(test_labels_i, axis=1)
train_label_original = np.argmax(test_data, axis=1)

'''
###NetML dataset####
print("\n Loading dataset NetML-2020......\n")
dataset = "/home/vibek/Anomanly_detection_packages/NetML-2020/data/NetML" 
anno = "top" # or "mid" or "fine"
submit = "both" # or "test-std" or "test-challenge"

# Assign variables
training_set = dataset+"/2_training_set"
training_anno_file = dataset+"/2_training_annotations/2_training_anno_"+anno+".json.gz"
test_set = dataset+"/1_test-std_set"
challenge_set = dataset+"/0_test-challenge_set"


# Get training data in np.array format
Xtrain, ytrain, class_label_pair, Xtrain_ids = get_training_data(training_set, training_anno_file)
print('Training class:\n', class_label_pair)

Xtest, ids = get_testing_data(test_set)

# Split validation set from training data
XX_train, X_vald, yy_train, y_vald = train_test_split(Xtrain, ytrain,
                                                test_size=0.2, 
                                                random_state=42,
                                                stratify=ytrain)

print("NetML train shape: ", XX_train.shape)
print("NetML train_label shape: ", yy_train.shape)
print("NetML validation shape: ", X_vald.shape)
print("NetML Validation_label shape: ", y_vald.shape)
print("NetML test shape: ", Xtest.shape)

# Preprocess the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(XX_train)
X_val_scaled = scaler.transform(X_vald)
X_test_scaled = scaler.transform(Xtest)
print("scaled value", X_val_scaled[:,0])

'''
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
tsne_results = tsne.fit_transform(train_data)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="dimension-1", y="dimension-2",
    hue="Label",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)

'''
tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    callbacks=ErrorLogger(),
    n_jobs=8,
    random_state=42,
)


embedding_train = tsne.fit(XX_train)

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)
sns.scatterplot(embedding_train[:,0], embedding_train[:,1], hue=yy_train, legend='full', palette=palette)
plt.xlabel("dimension-1")
plt.ylabel("dimension-2")
plt.show()


