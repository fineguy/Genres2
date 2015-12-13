# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:31:38 2015

@author: timasemenov
"""
import os
import glob
import time

import sunau
import numpy as np
import scipy
import scipy.io as sio
from matplotlib.pyplot import specgram

PATH_TO_MUSIC = ""


def file_iterator(base_dir=PATH_TO_MUSIC):
    genres = filter(lambda x: x[0] != '.', os.listdir(base_dir))

    for genre in genres:
        path_list = glob.glob(os.path.join(base_dir, genre + '/*.au'))
        for aufile_path in path_list:
            yield aufile_path


def get_data(aufile_path):
    ''' Opens .au file and return its frame rate and all frames '''
    aufile = sunau.open(aufile_path)
    sample_rate = aufile.getframerate()

    # read in frames as byte strings
    frames = aufile.readframes(aufile.getnframes())
    frames = np.fromstring(frames, dtype=np.int16)

    return (sample_rate, frames)


def plot_specgram(aufile_path):
    sample_rate, frames = get_data(aufile_path)
    specgram(frames, Fs=sample_rate, xextent=(0,30))


def create_fft(aufile_path):
    sample_rate, frames = get_data(aufile_path)
    fft_features = abs(scipy.fft(frames)[:1000])
    base_path, file_ext = os.path.splitext(aufile_path)
    data_path = base_path + ".fft"
    np.save(data_path, fft_features)


def read_fft(base_dir=PATH_TO_MUSIC):
    X = []
    y = []
    genres = filter(lambda x: x[0] != '.', os.listdir(base_dir))

    for label, genre in enumerate(genres):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)
        for file_path in file_list:
            fft_features = np.load(file_path)
            X.append(fft_features[:1000])
            y.append(label)

    return np.array(X), np.array(y)


def normalize(array):
    return array / array.max(axis=0)


def partition(array, parts):
    length = len(array) / parts * parts
    arr = np.mean(array[:length].reshape(-1, length/parts), axis=1)
    return normalize(arr)


def au2vec(aufile, duration):
    frames = aufile.readframes(aufile.getnframes())
    frames = np.fromstring(frames, dtype=np.int16)
    auvec = partition(frames, 20)

    aufft = np.fft.rfft(frames)
    aufft_abs = np.absolute(aufft)
    aufft_vec = partition(np.hstack((aufft.real, aufft.imag)), 20)
    aufft_abs_vec = partition(aufft_abs, 20)

    aufft_sq = np.fft.rfft([x*x for x in frames])
    aufft_sq_abs = np.absolute(aufft_abs)
    aufft_sq_vec = partition(np.hstack((aufft_sq.real, aufft_sq.imag)), 20)
    aufft_sq_abs_vec = partition(aufft_sq_abs, 20)

    return np.hstack((auvec, aufft_vec, aufft_abs_vec,
                      aufft_sq_vec, aufft_sq_abs_vec))


def save(X, y):
    sio.savemat('data.mat', {'X':X, 'y':y})


def main():
    start_time = time.time()

    for aufile_path in file_iterator():
        create_fft(aufile_path)

    end_time = time.time()
    print "Data processed in %.4f seconds" % (end_time - start_time)


if __name__ == "__main__":
    main()