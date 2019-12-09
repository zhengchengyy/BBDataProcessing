from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


# 导入全局变量
import GlobalVariable as gv
feature_names = gv.feature_names
# 删除名字后缀
feature_names = [feature[:-6] for feature in feature_names]
action_names = gv.action_names
color_list = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange', 'deepskyblue', 'pink']

# 定义特征阈值
# importance_threshold = 1 / len(feature_names)
importance_threshold = 0.1

# ————导入数据————
device_no = 2
feature_matrix = np.load('feature_matrixs/feature_matrix'+str(device_no)+'.npy')
label_matrix = np.load('feature_matrixs/label_matrix'+str(device_no)+'.npy')

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=0)

# 训练和预测
def train():
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('device_' + str(device_no) + '\'s test score:', test_score, round(test_score, 3))

    # 保存模型
    # import pickle
    # feature_num = feature_matrix.shape[1]
    # with open('models/' + 'device_' + str(device_no) + 'Acc_' + str(round(test_score, 3))
    #           + 'Fea_' + str(feature_num) + '.pickle', 'wb') as f:
    #     pickle.dump(clf, f)

    return clf

feature_importances = []
for i in range(len(feature_names)):
    feature_importances.append([])

for i in range(100):
    clf = train()
    temp = clf.feature_importances_
    for j in range(len(temp)):
        feature_importances[j].append(temp[j])
# print(feature_importances)

# 画出盒图
fig = plt.figure('device'+str(device_no)+'_feature_importance')
bplt = plt.boxplot(feature_importances, notch=False, sym='o', vert=True, patch_artist=True)
plt.axhline(y=importance_threshold, ls=":", color='r')
plt.xticks([x+1 for x in range(len(feature_names))], feature_names)
plt.xlabel('Features')  # x轴标题
plt.ylabel('Importance')  # y轴标题
plt.suptitle('Feature importances boxplot of device_'+str(device_no))  # 图形总标题
for pacthes, color in zip(bplt['boxes'], color_list):
    pacthes.set_facecolor(color)

plt.show()


# 按特征输入顺序画出条形图
def plot_feature_importances(feature_importance, feature_names):
    plt.bar(np.arange(feature_importance.shape[0]), feature_importance,
            tick_label=feature_names, color=color_list)
    plt.title('Feature importance')  # 子网格标题
    plt.xlabel('Features')  # x轴标题
    plt.ylabel('Importance')  # y轴标题
    plt.suptitle('Classification result')  # 图形总标题
    # 最大化显示图像窗口
    # plt.get_current_fig_manager().window.showMaximized()
    plt.show()  # 展示图形


# plot_feature_importances(feature_importance, feature_names)


# 按特征重要性排序
def plot_feature_importances_sorted(feature_importances, title, feature_names):
    feature_importances = 100 * (feature_importances / max(feature_importances))
    # 按特征重要性进行排序
    index_sorted = np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0]) + 0.8
    plt.figure()
    sorted_color_list = np.asarray(color_list)[index_sorted]
    plt.bar(pos, feature_importances[index_sorted], align='center', color=sorted_color_list)
    plt.xticks(pos, np.array(feature_names)[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


# plot_feature_importances_sorted(feature_importance, 'Feature importances', feature_names)
