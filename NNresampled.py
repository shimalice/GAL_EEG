import numpy as np
import scipy
import scipy.signal as signal
from scipy import fftpack
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from glob import glob
import os

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
%matplotlib inline

# LeaveOneGroupOut交差検定
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
# AUCスコア
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# keras
from keras.utils.np_utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import SGD

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

### データの読み込み ###

def prepare_data_train(fname):
    """ 訓練データの読み込み """
    # EEGデータ読み込み
    data = pd.read_csv(fname)
    # fnameイベントファイルの名前に変換
    events_fname = fname.replace('_data','_events')
    # イベントデータの読み込み
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#id列を削除
    labels=labels.drop(['id' ], axis=1)#id列を削除
    return  clean,labels

def prepare_data_test(fname):
    """ テストデータの読み込み """
    # EEGデータの読み込み
    data = pd.read_csv(fname)
    return data

### 前処理 ###

def preprocess_median_filter(X, kernel):
    """ Median filter"""
    X_m = signal.medfilt(X, kernel_size=kernel)
    return X_m

def preprocess_fir_filter(X, fc):
    """ FIR filter """
    fs = 500
    nyq = fs / 2.0  # ナイキスト周波数

    # フィルタの設計
    # ナイキスト周波数が1になるように正規化
    fe = fc / nyq      # カットオフ周波数1
    numtaps = 15          # フィルタ係数（タップ）の数（要奇数）

    b = scipy.signal.firwin(numtaps, fe) # Low-pass

    # FIRフィルタをかける
    X_FIR = scipy.signal.lfilter(b, 1, X)
    return X_FIR

def cut_off(X, fs):
    """ FFT処理後に高周波を取り除く"""
    # fs: カットオフ周波数[Hz]
    # 時系列のサンプルデータ作成
    n = X.shape[0]                         # データ数
    dt = 0.002                       # サンプリング間隔
    f = 500                           # 周波数

    # FFT 処理と周波数スケールの作成
    X_f = fftpack.fft(X)/(n/2)
    freq = fftpack.fftfreq(n, dt)

    # フィルタ処理
    # ここではカットオフ周波数以上に対応するデータを 0 にしている
    X_f2 = np.copy(X_f)
    X_f2[(freq > fs)] = 0
    X_f2[(freq < 0)] = 0

    # 逆 FFT 処理
    # FFT によるフィルタ処理では虚数部が計算されることがあるため
    # real 関数が必要(普段は必要ない)
    X_prep = np.real(fftpack.ifft(X_f2)*n)

    return X_prep


def data_preprocess_train(X):
    scaler= StandardScaler()
    X_prep = scaler.fit_transform(X)
#     X_prep = preprocess_fir_filter(X_prep, 100.0)
#     X_prep = cut_off(X, 50.0)

    #ここで他のpreprocessingを追加
    return X_prep

def data_preprocess_test(X):
    scaler= StandardScaler()
    X_prep = scaler.transform(X)
    #ここで他のpreprocessingを追加
    return X_prep

cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

#######number of subjects###############
subjects = range(1,13)
series = range(1, 9)
pred_tot = []
y_tot = []
global_auc = []
###loop on subjects and 8 series for train data + 2 series for test data
for i, subject in enumerate(subjects):
    print('Subject %d' % (subject))
    y_raw= []
    raw = []
    sequence = []
    auc_tot = []
    ################ READ DATA ################################################
    for ser in series:
        fname =  'input/train/subj%d_series%d_data.csv' % (subject,ser)
        data,labels=prepare_data_train(fname)
        raw.append(data)
        y_raw.append(labels)
        sequence.extend([ser]*len(data))

    X = pd.concat(raw)
    y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
    X = np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))
    y = y[:,0]
    sequence = np.asarray(sequence)
    X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X, y, test_size=0.3, random_state=71)

    # Under sampling [0]:[1] = 9:1
    positive_count_train = int(y_all_train.sum())
    rus = RandomUnderSampler(ratio={0:positive_count_train*9, 1:positive_count_train}, random_state=71)
    X_all_resampled, y_all_resampled = rus.fit_sample(X_all_train, y_all_train)
#     sequence_resampled, y_resampled = rus.fit_sample(sequence.reshape(-1, 1), y_all_train)

    # Over sampling [0]:[1] = 3:1
    ros = RandomOverSampler(ratio={0:X_all_resampled.shape[0], 1:X_all_resampled.shape[0]//3}, random_state=71)
    X_all_resampled, y_all_resampled = ros.fit_sample(X_all_resampled, y_all_resampled)

    # SMOTE sampling [0]:[1] = 5:1
#     smote = SMOTE(ratio={0:X_resampled.shape[0], 1:X_resampled.shape[0]//5}, random_state=71)
#     X_resampled, y_resampled = smote.fit_sample(X_resampled, y_resampled)

#     sequence_sorted = np.sort(sequence_resampled, axis=0)
#     sequence_resampled = sequence_sorted.flatten()
    y_all_binary = to_categorical(y_all_resampled)
    y_all_test = to_categorical(y_all_test)
#     print(sequence.shape, y.shape, X.shape)



    ########### Palameter ######################################################
    n_in = len(X[0])
    n_hiddens = [200, 200, 200]
    n_out = len(y_all_binary[0])
    p_keep = 0.5
    activation = 'relu'

    # model = Sequential()
    # for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
    #     model.add(Dense(n_hiddens[i], input_dim=input_dim))
    #     model.add(Activation(activation))
    #     model.add(Dropout(p_keep))

    # model.add(Dense(n_out))
    # model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=SGD(lr=0.00001),
    #               metrics=['accuracy'])

    epochs = 5
    batch_size = 200

    ################ Train classifiers ########################################
    cv = LeaveOneGroupOut()
#     cv.get_n_splits(groups=sequence_resampled)
    cvscores = []
    # pred = np.empty((X.shape[0],6))
    cvfold = 1
    auc_tot = []

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    for train, test in skf.split(X_all_resampled, y_all_resampled):
        print('\nFold: ', cvfold, '\n')
        cvfold = cvfold + 1

        X_train_resampled, X_test_resampled = X_all_resampled[train], X_all_resampled[test]
        y_train_resampled, y_test_resampled = y_all_binary[train], y_all_binary[test]
        #apply preprocessing
    #     X_train = data_preprocess_train(X_train)
    #     X_test = data_preprocess_test(X_test)

        model = Sequential()
        for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
            model.add(Dense(n_hiddens[i], input_dim=input_dim))
            model.add(BatchNormalization()) # バッチごとの白色化
            model.add(Activation(activation)) # ReLU
            model.add(Dropout(p_keep)) # Dropout

        model.add(Dense(n_out))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', # 交差entropy
                      optimizer=SGD(lr=0.01),
                      metrics=['accuracy'])
        model.fit(X_train_resampled, y_train_resampled, epochs=epochs,
                     batch_size=batch_size, verbose=1)

        scores = model.evaluate(X_test_resampled, y_test_resampled, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        scores_all = model.evaluate(X_all_test, y_all_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores_all[1]*100))
        cvscores.append(scores_all[1] * 100)
        pred = model.predict_proba(X_all_test)
        auc = roc_auc_score(y_all_test,pred)
        auc_tot.append(auc)

    auc_tot = np.asarray(auc_tot)

    print(auc_tot)
    print('Mean AUC: ', np.mean(auc_tot))
    global_auc.append(np.mean(auc_tot))

print("global_auc: ", sum(global_auc) / float(len(global_auc)), "\n")
print(global_auc, "\n")
print("global_acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)), "\n")
print(cvscores, "\n")
#     preds = Parallel(n_jobs=6)(delayed(predict)(clfs[i],X_test) for i in range(6))
#     pred[test,:] = np.concatenate(preds,axis=1)
