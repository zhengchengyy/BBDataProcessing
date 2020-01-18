from sklearn.tree import DecisionTreeClassifier
import numpy as np

ndevices = 5
start = 1
end = ndevices


def save_model(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix_bed' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix_bed' + str(device_no) + '.npy')
    # print(feature_matrix)

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    # 训练和预测
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('device_' + str(device_no) + '\'s train score:', train_score)
    print('device_' + str(device_no) +'\'s test score:', test_score)

    # 交叉验证
    from sklearn.model_selection import cross_val_score
    # print("交叉验证分数:", cross_val_score(clf, X_train, y=y_train, cv=5))  # cv表示几倍交叉验证
    print("交叉验证平均分数:", cross_val_score(clf, X_train, y=y_train, cv=10).mean())

    # 规则数
    tree_ = clf.tree_
    print("节点总数：", tree_.node_count)
    print("叶子数量：", tree_.n_leaves)

    # 保存模型
    import pickle
    feature_num = feature_matrix.shape[1]
    with open('models/' + 'device_' + str(device_no) + '_bed.pickle', 'wb') as f:
        pickle.dump(clf, f)

    # 导入全局变量
    import GlobalVariable as gv
    # action_names = gv.action_names
    action_names = ["get_up", "go_to_bed"]
    feature_names = gv.feature_names
    # 删除名字后缀
    feature_names = [feature[:-6] for feature in feature_names]

    # 决策树可视化
    from IPython.display import Image
    from sklearn import tree
    import pydotplus
    import os

    os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=action_names,
                                    filled=True, rounded=True,
                                    proportion=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf('trees/' + 'device_' + str(device_no) + 'Acc_' + str(round(test_score, 3))
    #            + 'Fea_' + str(feature_num) + '.pdf')
    graph.write_png('tree_images/' + 'device_' + str(device_no) + 'Acc_' + str(round(test_score, 3))
               + 'Fea_' + str(feature_num) + '_bed.png')

    # 直接画出决策树，但是很小看不清
    # from sklearn.tree import plot_tree
    # plot_tree(clf, filled=True)


for i in range(start, end + 1):
    save_model(i)
