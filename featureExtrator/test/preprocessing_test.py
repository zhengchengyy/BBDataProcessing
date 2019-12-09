import  numpy as np
from sklearn import preprocessing

# 归一化处理
a=np.array([1,2,3,4,5], dtype='float64')
print('a-1D:', a, a.shape)
a=a.reshape(-1,1)
print('a-2D:', a, a.shape)

print("z-score标准化")
# z-score标准化
a1 = preprocessing.scale(a)  #z-score标准化
print(a1)

print("z-score标准化")
# z-score标准化
a2 = preprocessing.StandardScaler().fit_transform(a)
print(a2)

print("min-max归一化")
# min-max归一化
a3 = preprocessing.MinMaxScaler().fit_transform(a)
print(a3)