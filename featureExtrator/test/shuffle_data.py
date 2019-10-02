import numpy as np


def shuffle_data(data, labels):
    #Shuffle data and labels.
    idx = np.arange(len(labels))  #和range差不多，但是支持步长为小数
    np.random.shuffle(idx)  #原地洗牌，直接改变值，而无返回值
    return data[idx, ...], labels[idx], idx

data = np.array([[1,2,3],[4,5,6],[7,8,9]])
labels = np.array([2,1,0])
# data, labels = shuffle_data(data, labels)
print(data[labels, ...])
print(labels)
