from glob import glob
import os
import numpy as np
import pandas as pd
from dataRead import prepare_data_train
from preprocess import data_preprocess_train
from powerSpectrum import spectrum
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

#被験者数
subjects = range(1,13)
seri
ids_tot = []
pred_tot = []

fnames = glob('input/train/subj1_series1_data.csv')
#print(fnames)
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


scores_str = ["%.4f" % x for x in scores_LR]
print(",".join(scores_str))

kf = StratifiedKFold(n_splits=10)
outcomes = []
fold = 0
for train_index, test_index in kf.split(X_Oz_f, y_f_handstart):
    clf = LogisticRegression()
    fold += 1
    X_train, X_test = X_Oz_f[train_index], X_Oz_f[test_index]
    y_train, y_test = y_f_handstart[train_index], y_f_handstart[test_index]
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    auc = roc_auc_score(y_test, predictions)
    outcomes.append(auc)
    print("Fold {0} auc: {1}".format(fold, auc))
mean_outcome = np.mean(outcomes)
print('mean_auc', mean_outcome, '\n')

# print('CV accuracy scores: %s' % scores_LR)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores_LR), np.std(scores_LR)))
# ind = np.arange(10)
# width = 0.35
# plt.figure()
# plt.bar(ind, scores_LR, width=width)
# plt.xlabel('Folds')
# plt.ylabel('ACC')
# plt.title('CV acc for each fold')
# plt.savefig('cross_val_acc_ex1.png' ,bbox_inches='tight')
