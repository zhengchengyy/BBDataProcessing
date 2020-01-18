from feature_modules import *
from feature_extractor import FeatureExtractor
import time

from pymongo import MongoClient
from exceptions import CollectionError
from pymongo import MongoClient

from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import os

from scipy import signal

device_no = 1

feature_matrix = np.load("feature_matrixs/feature_matrix" + str(device_no) + ".npy")
label_matrix = np.load("feature_matrixs/label_matrix" + str(device_no) + ".npy")


path = "feature_matrixs/feature_matrix" + str(device_no) + ".txt"
np.savetxt(path,feature_matrix)

for i in feature_matrix:
    print(feature_matrix)

# import pickle
# with open('models/' + 'device_' + str(device_no) + '_post_prune.pickle', 'rb') as f:
#     model = pickle.load(f)
#
# result = model.predict(feature_matrix)
# score = model.score(feature_matrix, label_matrix)
# for i in result:
#     print(i)
# print(result)
# print(score)

