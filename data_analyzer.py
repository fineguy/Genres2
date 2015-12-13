# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:41:50 2015

@author: timasemenov
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import os
import glob
from data_converter import PATH_TO_MUSIC


GENRES_OFFSET = ['classical', 'country', 'jazz', 'metal', 'pop', 'rock']

def read_fft(base_dir=PATH_TO_MUSIC):
    X = []
    y = []
    genres = filter(lambda x: x[0] != '.' and x in GENRES_OFFSET,
                    os.listdir(base_dir))

    for label, genre in enumerate(genres):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)
        for file_path in file_list:
            fft_features = np.load(file_path)
            X.append(fft_features[:1000])
            y.append(label)

    return np.array(X), np.array(y)


def plot_confusion_matrix(conf_mat, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(GENRES_OFFSET))
    plt.xticks(tick_marks, GENRES_OFFSET, rotation=45)
    plt.yticks(tick_marks, GENRES_OFFSET)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    X, y = read_fft()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(conf_mat)
    plt.figure()
    plot_confusion_matrix(conf_mat)

    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:,np.newaxis]
    print('Normalized confusion matrix')
    print(conf_mat_normalized)
    plt.figure()
    plot_confusion_matrix(conf_mat_normalized, title='Normalized confusion matrix')

    plt.show()


if __name__ == "__main__":
    main()