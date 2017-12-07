from glob import glob
import os
import numpy as np
import pandas as pd
from dataRead import prepare_data_train
from preprocess import data_preprocess_train
from powerSpectrum import spectrum
from sklearn.linear_model import LogisticRegression

cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

#被験者数
subjects = range(1,13)
ids_tot = []
pred_tot = []

fnames = glob('input/train/subj1_series1_data.csv')
print(fnames)
X = []
y = []
for fname in fnames:
    # 訓練データの読み込み
    data,labels = prepare_data_train(fname)
    # すべての系列データ(1-8)
    X.append(data)
    y.append(labels)

# 1次元のlistにする
X = pd.concat(X)
y = pd.concat(y)

#numpy arrayにする
X = np.asarray(X.astype(float))
y = np.asarray(y.astype(float))

X_Oz = X[:, 29]

point_cutoff = 40 + 1 #cutoff-frequency[Hz] * 2 + 1
X_Oz_f = np.empty((0,point_cutoff))
y_f_handstart = np.empty((0))

### t-500points(1sec)~t -> remove DC component -> power -> cut-off

for stride in range(500,len(X_Oz),3):
    """ stride:3, 150msec, 500Hz-> 1.5*500/3個の連続する1 """
    X_Oz_r = X_Oz[stride-500:stride]
    X_Oz_m = X_Oz_r - np.mean(X_Oz_r) # remove DC component
    dft_X_Oz_f = np.array(spectrum(X_Oz_m)[:point_cutoff]).reshape((1,-1)) # Power
    X_Oz_f = np.append(X_Oz_f, dft_X_Oz_f, axis=0)
    y_f_handstart = np.append(y_f_handstart, y[stride, 0])

# Standardize features by removing the mean and scaling to unit variance
X_Oz_f_s = data_preprocess_train(X_Oz_f)

clf = LogisticRegression()
scores_LR = cross_val_score(estimator=clf,
                         X=X_Oz_f,
                         y=y_f_handstart,
                         cv=10,
                         n_jobs=1)

print('CV accuracy scores: %s' % scores_LR)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores_LR), np.std(scores_LR)))
