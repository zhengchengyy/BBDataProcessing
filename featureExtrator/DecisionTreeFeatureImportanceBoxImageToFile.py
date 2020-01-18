# 画出特征重要性的盒图，注意！决策树的最大深度最好设置，防止过拟合
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# 导入全局变量
import GlobalVariable as gv

feature_names = gv.feature_names
# 删除名字后缀
feature_names = [feature[:-6] for feature in feature_names]
action_names = gv.action_names
color_list = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange', 'deepskyblue', 'pink']

device_no = 1
ndevices = 5

# 定义训练次数
train_times = 100
# 定义特征阈值
importance_threshold = 0.2


# 训练和预测
def save_importance_boxplot(device_no):
    # ————导入数据————
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    feature_importances = []
    for i in range(len(feature_names)):
        feature_importances.append([])

    for i in range(train_times):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        temp = model.feature_importances_
        for j in range(len(temp)):
            feature_importances[j].append(temp[j])

    # 画出盒图
    title = 'device' + str(device_no) + '_feature_importance'
    fig = plt.figure(title)
    bplt = plt.boxplot(feature_importances, notch=False, sym='.', vert=True, patch_artist=True)
    plt.axhline(y=importance_threshold, ls=":", color='r')
    plt.xticks([x + 1 for x in range(len(feature_names))], feature_names)
    plt.xlabel('Features')  # x轴标题
    plt.ylabel('Importance')  # y轴标题
    plt.suptitle('Feature importances boxplot of device_' + str(device_no))  # 图形总标题
    for pacthes, color in zip(bplt['boxes'], color_list):
        pacthes.set_facecolor(color)

    # plt.show()
    plt.savefig("feature_images/" + title + ".png")
    plt.close()


for device_no in range(1, ndevices + 1):
    save_importance_boxplot(device_no)
