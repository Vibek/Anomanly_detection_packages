from __future__ import division

import numpy as np
import pandas as pd
from matplotlib import pyplot
import itertools
import io
import tensorflow as tf
import datetime
import os
from scipy import stats

#import tensorflow
import h5py
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc, recall_score, f1_score)
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import make_scorer, auc
from sklearn.metrics import classification_report
import scikitplot as skplt
from sklearn.preprocessing import LabelEncoder 
from pathlib import Path

#custome libaries
#from data_preprocessing import NSL_kdd
from data_preprocessing_unsw import unsw_data
from Autoencoder_model import build_AE

# ***** REFERENCES PARAMETERS *****
params = {'batch_size': 1024, 
          'dataset': 'unsw-NB'}
result_path="/home/vibek/Anomanly_detection_packages/DNN_Package/"

#plot confusion matrix
def plot_confusion_matrix(cm, class_names):
  
    figure = pyplot.figure(figsize=(8, 8))
    pyplot.imshow(cm, interpolation='nearest', cmap=pyplot.cm.Blues)
    pyplot.title("Confusion matrix")
    pyplot.colorbar()
    tick_marks = np.arange(len(class_names))
    pyplot.xticks(tick_marks, class_names, rotation=45)
    pyplot.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        pyplot.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.show()
    return figure

#plot ROC graph
def plot_roc_curve(Y_test, Y_pred, class_name, class_index, title='ROC'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_name):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    #plt.figure()
    lw = 2
    pyplot.plot(fpr[class_index], tpr[class_index], color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[class_index])
    pyplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title(title)
    pyplot.legend(loc="lower right")
    pyplot.show()


#label and category accuracy
def to_cat(y):
    y_tmp = np.ndarray(shape=(y.shape[0], 2), dtype=np.float32)
    for i in range(y.shape[0]):
        y_tmp[i, :] = np.array([1-y[i], y[i]])   # np.array([0,1]) if y[i] else np.array([1,0])
    return y_tmp

#Save labels of each category 
def save_labels(predic, actual, result_path, phase, accur):
        labels = np.concatenate((predic, actual))

        
        if not os.path.exists(path=result_path):
            os.mkdir(path=result_path, exist_ok=True)

        np.save(file=os.path.join(result_path, '{}-DNN-Results-{}.npy'.format(phase, float('%.2f'%score))), arr=labels)


with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
	model_path = "/home/vibek/Anomanly_detection_packages/DNN_Package/Model_DNN/"
	model_name = model_path+"best_model_final.hdf5"
	model = load_model(model_name, compile=False)
	model.summary()
# try using different optimizers and different optimizer configs	
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#Load dataset
train_data, train_labels, test_data, test_labels = unsw_data(params)

#Print information of the results
print('Val loss and acc:\n')

 # Model for testing
print(model.evaluate(test_data, test_labels, batch_size=512, verbose=0))

np.savetxt("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/model_evaluate_1.csv", model.evaluate(test_data, test_labels, verbose=1), delimiter=",")
#Print information of the results
print('Predict_result:\n')

# Model for prediction: Used to just return predicted values

test_pred_raw = model.predict(test_data, batch_size=512, verbose=0)
test_pred = np.argmax(test_pred_raw, axis=1)
test_label_original = np.argmax(test_labels, axis=1)
test_mae_loss = np.mean(np.power(test_data - test_pred[:, np.newaxis], 2), axis=1)
print('error_test:', test_mae_loss)
#print('test_label_original:', test_label_original)
#print('test_pred:', test_pred)

train_pred_raw = model.predict(train_data, batch_size=512, verbose=0)
train_pred = np.argmax(train_pred_raw, axis=1)
train_label_original = np.argmax(test_labels, axis=1)
train_mae_loss = np.mean(np.abs(train_pred[:, np.newaxis] - train_data), axis=1)
print('error_train:', train_mae_loss)


np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test.npy', test_pred)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_test.npy', test_label_original)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_train.npy', train_pred)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_train.npy', train_label_original)
np.savetxt("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/Predict_result_1.csv", test_pred, delimiter=",")

 # To encode string  labels into numbers
score = accuracy_score(test_label_original, test_pred)
print("Accuracy_score:", score)

print('Classification Report:\n')
print(classification_report(test_label_original, test_pred))

#Anomaly score
error_df = pd.DataFrame({'reconstruction_error': test_mae_loss,'true_class': test_label_original})
anomaly_error_df = error_df[error_df['true_class'] == 1]
print('Detection Result:', error_df)
print('Anomaly Score:', error_df.describe())

 #Visualize confusion matrix
print('\nConfusion Matrix:')
conf_matrix = confusion_matrix(test_label_original, test_pred)
print(conf_matrix)
figure = plot_confusion_matrix(conf_matrix, class_names=list(range(2))) 
save_labels(predic=test_pred, actual=test_label_original, result_path=result_path, phase='testing', accur=score)

# Log the confusion matrix as an image summary.
plot_roc_curve(to_cat(test_label_original), to_cat(test_pred), 2, 0, title='Receiver operating characteristic (attack_or_not = 0)')
plot_roc_curve(to_cat(test_label_original), to_cat(test_pred), 2, 1, title='Receiver operating characteristic (attack_or_not = 1)')

FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  # False Positive
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)  # False Negative
TP = np.diag(conf_matrix)  # True Positive
TN = conf_matrix.sum() - (FP + FN + FP)  # True Negative

print('\nTPR:')  # True Positive Rate
# Portion of positive instances correctly predicted positive
print(TP / (TP + FN))
np.savetxt("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/True_positive_rate_1.csv", TP / (TP + FN), delimiter=",")

print('\nFPR:')  # False Positive Rate
# Portion of negative instances wrongly predicted positive
print(FP / (FP + TN))
np.savetxt("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/False_positive_rate_1.csv", FP / (FP + TN), delimiter=",")

# Cost Matrix as presented in Staudemeyer article
cost_matrix = [[0, 1, 2, 2, 2],
               [1, 0, 2, 2, 2],
               [2, 1, 0, 2, 2],
               [4, 2, 2, 0, 2],
               [4, 2, 2, 2, 0]]

cost_matrix = [[0, 1],
               [1, 0]]

tmp_matrix = np.zeros((2, 2))

for i in range(2):
	for j in range(2):
		tmp_matrix[i][j] = conf_matrix[i][j] * cost_matrix[i][j]

# The average cost is (total cost / total number of classifications)
print('\nCost:')
print(tmp_matrix.sum()/conf_matrix.sum())

print('\nAUC:')  # Average Under Curve
print(roc_auc_score(y_true=test_label_original, y_score=test_pred, average='macro'))

#pre-defined threshold
threshold=0.75

#Reconstruction error score
anomalies = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

groups = error_df.groupby('true_class')
fig, ax = pyplot.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=2, linestyle='',
            label= "Anomaly" if name == 1 else "Normal", color= 'red' if name == 1 else 'black')
ax.legend()
pyplot.title("Reconstruction error score")
pyplot.ylabel("Reconstruction error")
pyplot.xlabel("Data point index")
pyplot.show();

#Plot anomalies graph to visualize the experimental results
s1 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_test.npy')
s2 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test.npy')

test_pred_plot = s2 > 0.7

index_same = np.argwhere(s1 == test_pred_plot)
index_diff = np.argwhere(s1 != test_pred_plot)

normal = np.where(s1 == test_pred_plot)
anomaly = np.where(s1 != test_pred_plot)

dt = 1.0
t = np.arange(0, len(s1), dt)
s3 = np.ones(len(s1)) * 0.5
#print('The value of s3:', s3, t)

#plot the anomalies result figure
fig = pyplot.figure(1)
ax = fig.add_subplot(111)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

#classification results
ax.plot(t, s1, markersize=5, label='True Label', color='black', marker='*', linestyle='-')
ax.plot(t[index_same][:,0], test_pred_plot[index_same][:,0], markersize=3, label='correctly predicted', color='blue', marker='*', linestyle='')
ax.plot(t[index_diff][:,0], test_pred_plot[index_diff][:,0], markersize=3, label='wrongly predicted', color='red', marker='*', linestyle='')

#ax.plot(t, s3, 'r-')
#ax.set_ylim(-0.3, 1.5)
ax.set_xlabel('Number of Samples', font2)
ax.set_ylabel('Probability of Each Sample', font2)
ax.legend(loc='upper right', prop=font2)
pyplot.title('Classification Result', font1)
ax.grid(True)
pyplot.show()

# Draw the wrongly detected result values
fig2 = pyplot.figure(1)
ax1 = fig2.add_subplot(111)
count_line = np.zeros(len(s1))
index_low = 0
index_high = 0
for i, index in enumerate(index_diff):

    index_high = index[0]
    count_line[index_low:index_high] = i
    index_low = index_high
count_line[81918:] = 1506
ax1.plot(t, count_line)
pyplot.title('Cumulative Amount of Incorrect Detection', font1)
ax1.set_xlabel('Number of Samples', font2)
ax1.set_ylabel('Number of Incorrect Detection', font2)

pyplot.subplots_adjust(wspace=0., hspace =0.3)
pyplot.show()

# Create outlier detection plot
fig = pyplot.figure(1)
ax2 = fig.add_subplot(111)

ax2.scatter(normal, test_data[normal][:,0], c= 'blue', marker='*', label='Normal', s=1)
ax2.scatter(anomaly,test_data[anomaly][:,0], c= 'red', marker='*', label='Anomaly', s=5)
pyplot.title('Anomaly Detection')
pyplot.legend(loc=2)
pyplot.show()


"""
class anomaly_detect():
  
        def __init__(self,method='average',window=5,max_outliers=None,alpha=0.05,mode='same'):
        self.method = method
        self.window = window
        self.max_outliers = max_outliers
        self.alpha = alpha
        self.mode = mode

        def moving_average(self,f_t):
        if type(f_t) is not np.ndarray:
            raise TypeError\
                ('Expected one dimensional numpy array.')
        if f_t.shape[1] != 1:
            raise IndexError\
                ('Expected one dimensional numpy array, %d dimensions given.' % (f_t.shape[1]))

        f_t = f_t.flatten()
        window = self.window
        mode = self.mode
        g_t = np.ones(int(window))/float(window)
        # Deal with boundaries with atleast lag/2 day window
        #mode = 'same'
        rolling_mean = np.convolve(f_t,g_t,mode)
        self.rolling_mean = rolling_mean
        return rolling_mean

        def deviation_stats(self,df):
        df['mean_count'] = self.rolling_mean
        df['residual'] = df.iloc[:,0] - self.rolling_mean
        std_resid = np.std(df.residual)
        df['pos_std'] = df.mean_count + std_resid
        df['neg_std'] = df.mean_count - std_resid
        df['pos_std_2'] = df.mean_count + 2*std_resid
        df['neg_std_2'] = df.mean_count - 2*std_resid
        return df

        def normality(self):
        
        Plots the distribution and probability plot for the Residuals
        These two plots are used for a sanity check to confirm that the
        residual between the actual data and the moving average are
        approximately normally distributed.
        This is important as the ESD test can only be used if the data is
        approximately normally distributed.  Refer to notes and References
        for more details.
        
        if self.results is not None:
            df = self.results
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6))
            x = df.residual.values
            re = stats.probplot(x, plot=ax2)
            ax1.hist(df.residual,bins=100);
            ax1.set_title('Distribution of Residuals');
        else:
            raise NameError\
                ('The moving average for the data has not yet been computed.  Run moving_averge or evaluate prior to normality.')

"""
