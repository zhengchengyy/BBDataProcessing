import numpy as np

feature_matrix = np.load('feature_matrixs/feature_matrix2.npy')
label_matrix = np.load('feature_matrixs/label_matrix2.npy')

train_size = feature_matrix.shape[0] // 5 * 4
test_size = feature_matrix.shape[0] - train_size

trainfea_matrix = feature_matrix[0:train_size]
trainlab_matrix = label_matrix[0:train_size]
test_fea_matrix = feature_matrix[train_size:]
test_lab_matrix = label_matrix[train_size:]

# print(trainfea_matrix)

print(trainfea_matrix.shape)

# print(test_fea_matrix)

print(test_fea_matrix.shape)

print(feature_matrix.shape)

print(feature_matrix)

print(label_matrix.shape)

print(label_matrix)

# np.savetxt('feature_matrixs/feature_matrixs.txt',feature_matrix)
# np.savetxt('feature_matrixs/label.txt',label_matrix)
