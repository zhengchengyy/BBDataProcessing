from sklearn.tree import DecisionTreeClassifier
import numpy as np

# ndevices = 3
# start = 2
# end = 2

ndevices = 3
start = 1
end = ndevices

def save_model(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.25, random_state=0)

    # 训练和预测
    # clf = DecisionTreeClassifier(random_state=0,
    #                              max_depth=13,
    #                              max_leaf_nodes=25,
    #                              min_impurity_decrease=0.0001,
    #                              min_samples_leaf=3,
    #                              min_samples_split=3,
    #                              splitter='best',
    #                              criterion='gini')
    clf = DecisionTreeClassifier(random_state=0,
                                 max_depth=13,
                                 max_leaf_nodes=24,
                                 min_impurity_decrease=0.0003,
                                 min_samples_leaf=3,
                                 min_samples_split=7,
                                 splitter='best',
                                 criterion='entropy')
    # clf = DecisionTreeClassifier(random_state=0,
    #                              max_depth=11,
    #                              max_leaf_nodes=24,
    #                              min_impurity_decrease=0.0001,
    #                              min_samples_leaf=3,
    #                              min_samples_split=2,
    #                              splitter='best',
    #                              criterion='entropy')
    # clf = DecisionTreeClassifier(random_state=0,
    #                              max_depth=13,
    #                              max_leaf_nodes=31,
    #                              min_impurity_decrease=0.0001,
    #                              min_samples_leaf=4,
    #                              min_samples_split=3,
    #                              splitter='best',
    #                              criterion='entropy')
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('device_' + str(device_no) + '\'s train score:', train_score, round(train_score,3))
    print('device_' + str(device_no) +'\'s test score:', test_score, round(test_score,3))

    # 保存模型
    import pickle
    feature_num = feature_matrix.shape[1]
    with open('models/' + 'device_' + str(device_no) + 'Acc_' + str(round(test_score, 3))
               + 'Fea_' + str(feature_num) + '_prune.pickle', 'wb') as f:
        pickle.dump(clf, f)

    # 导入全局变量
    import GlobalVariable as gv
    action_names = gv.action_names
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
                                    impurity=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf('trees/' + 'device_' + str(device_no) + 'Acc_' + str(round(test_score, 3))
    #            + 'Fea_' + str(feature_num) + '.pdf')
    graph.write_png('tree_images/' + 'device_' + str(device_no) + 'Acc_' + str(round(test_score, 3))
               + 'Fea_' + str(feature_num) + '_prune.png')

    # 直接画出决策树，但是很小看不清
    # from sklearn.tree import plot_tree
    # plot_tree(clf, filled=True)


for i in range(start, end + 1):
    save_model(i)
