import numpy as np

feature_matrix = np.asarray([1, 2, 3])
np.save('test', feature_matrix)

feature_matrix2 = np.load('test.npy')
feature_matrix3 = np.append(feature_matrix2, [4])
# feature_matrix3 = np.append(feature_matrix2, [4], axis=0)
np.save('test', feature_matrix3)
feature_matrix4 = np.load('test.npy')

print("feature_matrix：", feature_matrix)
print("feature_matrix2：", feature_matrix2)
print("feature_matrix3：", feature_matrix3)
print("feature_matrix4：", feature_matrix4)