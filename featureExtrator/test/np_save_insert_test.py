import numpy as np
import os

feature_matrix = np.asarray([[1, 2, 3]])
np.save('test', feature_matrix)

feature_matrix = np.load('test.npy')
print("insert before\n", feature_matrix)

os.remove("test.npy")

temp_matrix = [2, 3, 4]
feature_matrix = np.insert(feature_matrix, feature_matrix.shape[0],
                                           values=temp_matrix, axis=0)

print("insert after\n", feature_matrix)
np.save('test', feature_matrix)


feature_matrix = np.load('test.npy')
print("save after\n", feature_matrix)


os.remove("test.npy")

temp_matrix = [3, 4, 5]
feature_matrix = np.insert(feature_matrix, feature_matrix.shape[0],
                                           values=temp_matrix, axis=0)
np.save('test', feature_matrix)
feature_matrix = np.load('test.npy')
print("insert second time\n", feature_matrix)
