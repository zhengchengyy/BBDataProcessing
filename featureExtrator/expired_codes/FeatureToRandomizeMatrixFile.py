import numpy as np


def shuffle_data(data, labels):
    # Shuffle data and labels.
    idx = np.arange(len(labels))  # 和range差不多，但是支持步长为小数
    np.random.shuffle(idx)  # 原地洗牌，直接改变值，而无返回值
    return data[idx, ...], labels[idx], idx

ndevices = 3
for i in range(1, ndevices+1):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(i) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(i) + '.npy')

    # 随机打乱数据
    feature_matrix, label_matrix, _ = shuffle_data(feature_matrix, label_matrix)

    # 保存文件
    np.save('feature_matrixs/feature_random_matrix' + str(i), feature_matrix)
    np.save('feature_matrixs/label_random_matrix' + str(i), label_matrix)
