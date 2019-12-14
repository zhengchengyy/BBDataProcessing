from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier


# 导入全局变量
import GlobalVariable as gv
feature_names = gv.feature_names
# 删除名字后缀
feature_names = [feature[:-6] for feature in feature_names]
action_names = gv.action_names

# ————导入数据————
device_no = 2
feature_matrix = np.load('feature_matrixs/feature_matrix'+str(device_no)+'.npy')
label_matrix = np.load('feature_matrixs/label_matrix'+str(device_no)+'.npy')

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=0)


rf = RandomForestClassifier(n_estimators=20)

def train():
    scores = []
    for i in range(X_train.shape[1]):
         # 每次选择一个特征，进行交叉验证，训练集和测试集为7:3的比例进行分配，
         # ShuffleSplit()函数用于随机抽样（数据集总数，迭代次数，test所占比例）
         # cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
         # cv = ShuffleSplit(len(X_train), 3, .3)
         score = cross_val_score(rf, X_train[:, i:i+1], y_train, cv=10)
         # scoring="r2"表示拟合优度(Goodness of Fit)是指回归直线对观测值的拟合程度。
         # 度量拟合优度的统计量是可决系数(亦称确定系数)R²。R²最大值为1。
         # R²的值越接近1，说明回归直线对观测值的拟合程度越好；反之，R²的值越小，说明回归直线对观测值的拟合程度越差。
         # https://blog.csdn.net/Softdiamonds/article/details/80061191
         scores.append((round(np.mean(score), 3), feature_names[i]))
    print(sorted(scores, reverse=True))

for i in range(10):  #多次训练得到的特征分数差不多，相差不超过0.01
    train()