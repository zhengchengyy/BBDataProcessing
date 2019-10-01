import numpy as np

feature_matrix = np.load('feature_matrix2.npy', allow_pickle=True)

print(feature_matrix.shape)

print(feature_matrix)


label_matrix = np.load('label_matrix2.npy', allow_pickle=True)

print(label_matrix.shape)

print(label_matrix)

np.savetxt('feature_matrix.txt',feature_matrix)
np.savetxt('label.txt',label_matrix)
