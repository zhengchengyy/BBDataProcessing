import numpy as np
from math import sqrt
from sklearn import tree
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF
import copy
import pydotplus
import os

# 导入数据
device_no = 1
feature_matrix = np.load('feature_matrixs/feature_matrix_bed' + str(device_no) + '.npy')
label_matrix = np.load('feature_matrixs/label_matrix_bed' + str(device_no) + '.npy')
# print(feature_matrix)

# 定义训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=0)
print("训练集长度:", len(X_train), len(y_train))
print("测试集长度：", len(X_test), len(y_test))

# feature_names = ["MeanModule", "SDModule"]
# class_names = ["turn_over", "legs_stretch", "hands_stretch",
#                "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

# 导入全局变量
import GlobalVariable as gv
# class_names = gv.action_names
class_names = ["get_up", "go_to_bed"]
feature_names = gv.feature_names

# 删除名字后缀
feature_names = [feature[:-6] for feature in feature_names]

def model_json():
    # 9动作2特征
    clf = DecisionTreeClassifier(random_state=0,
                                 max_depth=11,
                                 max_leaf_nodes=78,
                                 min_impurity_decrease=0.00032,
                                 min_samples_leaf=2,
                                 min_samples_split=5,
                                 splitter='best',
                                 criterion='entropy')
    # clf = DecisionTreeClassifier(random_state=0,
    #                              max_depth=13,
    #                              max_leaf_nodes=24,
    #                              min_impurity_decrease=0.0003,
    #                              min_samples_leaf=3,
    #                              min_samples_split=7,
    #                              splitter='best',
    #                              criterion='entropy')

    # 直接后剪枝
    # clf = DecisionTreeClassifier(random_state=0,
    #                              max_depth=20,
    #                              criterion='entropy')

    print("now training,wait please..........")
    clf.fit(X_train, y_train)
    print("train finished")
    result = rules(clf, feature_names, class_names)

    # 保存为json文件
    # with open('structure.json', 'w') as f:
    #     f.write(json.dumps(result))
    print("The json-style model has been stored in structure.json")

    print("now I'm drawing the CART tree,wait please............")

    dot_file = "visualization/T0.dot"
    png_file = "visualization/T0.png"
    draw_file(clf, dot_file, png_file, feature_names, class_names)
    print("CART tree has been drawn in " + png_file)
    return clf, result


def binaryTreePaths(model, root, feature_names, action_names):
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    if root == _tree.TREE_UNDEFINED:
        return []
    res, stack = [], [(root, "")]
    node_num = 0
    while stack:
        node, ls = stack.pop()
        node_num += 1
        left = tree_.children_left[node]
        right = tree_.children_right[node]
        val = str(round(tree_.threshold[node], 3))
        name = feature_name[node]
        if left == _tree.TREE_LEAF and right == _tree.TREE_LEAF:
            idx = np.argmax(tree_.value[node])
            result = " THEN action = " + action_names[idx]
            res.append("IF " + ls + val + result)
        if right != _tree.TREE_LEAF:
            stack.append((right, ls + name + " > " + val + " AND "))
        if left != _tree.TREE_LEAF:
            stack.append((left, ls + name + " <= " + val + " AND "))
    return res, node_num


def rules(clf, features, labels, node_index=0):
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # 叶子节点
        count_labels = list(zip(clf.tree_.value[node_index, 0], labels))
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  # 所谓的class_name其实就是在这种地方用到了，这个class_names其实可以理解为类别的取值
                                  for count, label in count_labels))

        node['value'] = [count for count, label in count_labels]  # add by appleyuchi
    else:

        count_labels = list(zip(clf.tree_.value[node_index, 0], labels))  # add by appleyuchi
        node['value'] = [count for count, label in count_labels]  # add by appleyuchi

        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} <= {}'.format(feature, threshold)
        left_index = clf.tree_.children_right[node_index]
        right_index = clf.tree_.children_left[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node


def draw_file(model, dot_file, png_file, feature_names, class_names):
    os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'
    dot_data = tree.export_graphviz(model, out_file=dot_file,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)

    graph = pydotplus.graph_from_dot_file(dot_file)
    graph.write_png(png_file)

    # 原始代码生成svg文件，用浏览器打开的图片文件
    # thisIsTheImage = Image(graph.create_png())
    # display(thisIsTheImage)
    # print(dt.tree_.feature)
    # from subprocess import check_call
    # check_call(['dot', '-Tsvg', dot_file, '-o', png_file])


def classify(json_model, feature_names_list, test_data):
    if "children" not in json_model:
        return json_model["value"]  # 到达叶子节点，完成测试

    bestfeature = json_model["name"].split("<=")[0].strip()
    threshold = float(json_model["name"].split(bestfeature + " <= ")[1].strip())
    test_best_feature_value = test_data[feature_names_list.index(bestfeature)]
    if float(test_best_feature_value) <= threshold:
        child = json_model["children"][0]
        result = classify(child, feature_names_list, test_data)
    else:
        child = json_model["children"][1]
        result = classify(child, feature_names_list, test_data)

    return result


def Tt_count(model, count):  # |Tt|
    if "children" not in model:
        return 1
    children = model["children"]
    for child in children:
        count += Tt_count(child, 0)
    return count


def Rt_compute(model):
    # R(t)注意，这个地方我们使用的是错误数量，依据是《Simplifying Decision Trees》-quinlan
    # 也可以改成错误率，依据是ＣＣＰ的原文《classification and regression trees》-Leo Breiman
    # 这两篇参考文献都是决策树的发明者写的。
    Rt = (sum(model['value']) - max(model['value']))
    return Rt


def RTt_compute(model, leaves_error_count):
    # R(Tt)
    # this function is modified from the above function "Tt_count"
    if "children" not in model:
        return Rt_compute(model)
    children = model["children"]
    for child in children:
        leaves_error_count += RTt_compute(child, 0)
    return leaves_error_count


def gt_compute(model):
    return (Rt_compute(model) - RTt_compute(model, 0) * 1.0) / (Tt_count(model, 0) - 1)


#       R(t)-R(Tt)
# g(t)=-----------
#       |Tt|-1
# T0->a0
# T1->a1
# T2->a2
# T3->a3
# ···
# model=T0
# for example,to get T1,we need to know which node of T0 has the minimum g(t)


# T0->T1
# 根据g(t)最小获得pruned_parts（也就是要裁剪的部分），然后对当前模型进行剪枝。
def T1_create(model, gt_list, prune_parts, prune_gt_index):  # 完成,这个函数遍历了每一个节点
    if 'children' not in model:  # 如果是叶子节点
        return
    else:  # 加上当前节点的gt值以及保存当前“假设要裁掉”的树部分
        gt_list.append(gt_compute(model))
        prune_parts.append(model)
        children = model["children"]
        if len(prune_parts) == prune_gt_index + 1:
            del model["children"]
        for child in children:
            T1_create(child, gt_list, prune_parts, prune_gt_index)


# ☆☆☆☆☆☆☆modify sklearn-model synchronized with json-model☆☆☆☆☆☆☆☆
def prune_sklearn_model(sklearn_model, index, json_model):
    if "children" not in json_model:  # json_model is the noly node of the tree.
        sklearn_model.children_left[index] = TREE_LEAF
        sklearn_model.children_right[index] = TREE_LEAF
    else:
        prune_sklearn_model(sklearn_model, sklearn_model.children_left[index], json_model["children"][0])
        prune_sklearn_model(sklearn_model, sklearn_model.children_right[index], json_model["children"][1])


def gt_with_tree(model, gt_list, prune_parts):  # 完成,这个函数遍历了每一个节点
    if 'children' not in model:  # 如果是叶子节点
        return
    else:  # 加上当前节点的gt值以及保存当前“假设要裁掉”的树部分
        gt_list.append(gt_compute(model))
        prune_parts.append(model)
        children = model["children"]
        for child in children:
            gt_with_tree(child, gt_list, prune_parts)


def predict(json_model, feature_names, class_names, test_item):
    # print("进入predict时的测试数据",test_item)
    leaf_value = classify(json_model, feature_names, test_item)
    class_names_index = leaf_value.index(max(leaf_value))
    result = class_names[class_names_index]
    return result, class_names_index


def precision_compute(json_model, X_test, y, feature_names, class_names):
    count_right = 0.0
    X_test = np.array(X_test)
    X_test = X_test.tolist()
    # -----------------------
    for index, item in enumerate(X_test):
        predict_result, class_name_index = predict(json_model, feature_names, class_names, item)
        # if class_names[class_name_index] == str(y[index]):  #原代码比较的是名称
        if class_name_index == y[index]:
            count_right += 1
    #     else:
    #         print"测试失败数据为：",item
    #         print"实际结果为",y[index]
    #         print"预测结果为",class_names[class_name_index]
    # print"测试准确的有:",count_right
    return count_right / len(X_test)


# here model is json-style model
def model_gtmin_Tt(clf, model, feature_names, class_names, Tt_name):  # T0->T1
    Tt = Tt_count(model, 0)  # |Tt|
    Rt = Rt_compute(model)
    RTt = RTt_compute(model, 0)

    gt_list = []
    prune_parts = []
    gt_with_tree(model, gt_list, prune_parts)

    alpha = min(gt_list)
    prune_gt_index = gt_list.index(alpha)
    prune_for_minimum_gt = prune_parts[prune_gt_index]  #

    T0 = copy.deepcopy(model)
    T1 = copy.deepcopy(model)  # here T1 means Ti
    gt_list = []  # 这里必须复位清零
    prune_parts = []  # 这里必须复位清零
    T1_create(T1, gt_list, prune_parts, prune_gt_index)
    # 这里不使用上面的prune_for_minimum的原因是，这个被裁掉的部分，你不知道处于哪个结点下面．
    # 也就是说，你虽然知道要裁掉的子树是什么，但是你无法知道在哪里裁，所以这里对prune_parts进行重新构建
    # from T0(original model) to get T1
    # print"\nT0=",model
    # print "\nT1=",T1

    index = 0  # never change this value！！！
    sklearn_model = copy.deepcopy(clf)
    prune_sklearn_model(sklearn_model.tree_, index, T1)
    dot_file = "visualization/T" + Tt_name + ".dot"
    png_file = "visualization/T" + Tt_name + ".png"
    # draw_file(sklearn_model,dot_file,png_file,X_train,feature_names)
    draw_file(sklearn_model, dot_file, png_file, feature_names, class_names)
    return sklearn_model, T1, alpha


def CCP_TreeCandidate(clf, current_model, feature_names, class_names, alpha_list,
                      Ti_list):  # get the tree Sets with each minimum "g(t)"
    Flag = True
    alpha = 0
    Tt_name = 0
    scikit_model = copy.deepcopy(clf)
    current_json_model = copy.deepcopy(current_model)
    # print("current_json_model=", current_json_model)

    while Flag:
        alpha_list.append(alpha)
        Ti_list.append(current_json_model)
        print("We have gotten the final T" + str(Tt_name) + ", please wait.......")

        Tt_name = Tt_name + 1
        scikit_model, current_json_model, alpha = model_gtmin_Tt(scikit_model, current_json_model,
                                                                 feature_names, class_names, str(Tt_name))

        if "children" not in current_json_model:  # only root node
            print("截止条件中的current_json_model=", current_json_model)
            Ti_list.append(copy.deepcopy(current_json_model))
            alpha_list.append(alpha)
            Flag = False
            print("We have gotten the final Ti")

    return alpha_list, Ti_list


def CCP_validation(TreeSets, alpha_list, X_test, y_test, feature_names, class_names, sklearn_model, b_SE):
    precision_list = []
    train_score_list = []
    progress_length = len(TreeSets)

    for index, item in enumerate(TreeSets):
        Ti_precision = precision_compute(item, X_test, y_test, feature_names, class_names)
        print("T%d_precision(test_score)=%f" % (index, Ti_precision))

        train_score = precision_compute(item, X_train, y_train, feature_names, class_names)
        print("T%d_precision(train_score)=%f" % (index, train_score))
        train_score_list.append(train_score)

        precision_list.append(Ti_precision)
        print("the T" + str(index) + " has been validated, " + str(
            progress_length - index - 1) + " Trees left, please wait.....")
    if b_SE == False:  # 直接选择精确率最大的树作为后剪枝最佳的树
        pruned_precision = max(precision_list)
        index = precision_list.index(pruned_precision)
        print("index=", index)
        best_alpha = alpha_list[index]
        Best_tree = TreeSets[index]
        dot_file = "visualization/Best_tree_0SE.dot"
        png_file = "visualization/Best_tree_0SE.png"
        # 画一画树
        best_sklearn_model = copy.deepcopy(sklearn_model)
        prune_sklearn_model(best_sklearn_model.tree_, 0, Best_tree)

        draw_file(best_sklearn_model, dot_file, png_file, feature_names, class_names)
        return Best_tree, best_alpha, pruned_precision, precision_list[0]

    else:  # 使用1-SE rule，直接选择误差率最低的树作为后剪枝最佳的树
        error_rate_list = [1 - item for item in precision_list]
        lowest_error_rate = min(error_rate_list)
        print("error_rate_list=", error_rate_list)
        print("precision_list=", precision_list)
        SE = sqrt(lowest_error_rate * (1 - lowest_error_rate) / len(y_test))
        print("SE=", SE)

        criterion_1_SE = lowest_error_rate + SE

        index_error_rate = 0
        # search from from the end ,because the error_rate_list is not monotory.
        for index, item in enumerate(error_rate_list):
            if error_rate_list[len(error_rate_list) - 1 - index] < criterion_1_SE:
                index_error_rate = len(error_rate_list) - 1 - index
                break

        # if index_error_rate-1>=0:
        #     index_error_rate=index_error_rate-1
        # else:
        #     pass#becasuse the list may only have one item.

        # here's right,because the precision list is corresponding to the error_rate_list.
        pruned_precision = precision_list[index_error_rate]
        train_pruned_precision = train_score_list[index_error_rate]

        best_alpha = alpha_list[index_error_rate]
        Best_tree = TreeSets[index_error_rate]
        dot_file = "visualization/Best_tree_1SE.dot"
        png_file = "visualization/Best_tree_1SE.png"
        # 画一画树
        best_sklearn_model = copy.deepcopy(sklearn_model)
        original_tree = best_sklearn_model.tree_
        print("原始节点总数：", original_tree.node_count)
        print("原始叶子数：", original_tree.n_leaves)

        prune_sklearn_model(best_sklearn_model.tree_, 0, Best_tree)

        draw_file(best_sklearn_model, dot_file, png_file, feature_names, class_names)

        # 下面代码不可用，因为sklearn模型不能改变原来的节点数和叶子数属性
        # current_tree = best_sklearn_model.tree_
        # print("Current_tree's node num = ", current_tree.node_count)
        # print("Current_tree's leaf num = ", leaf_num)

        # 提取后剪枝后的规则
        res, node_num = binaryTreePaths(best_sklearn_model, 0, feature_names, class_names)
        print("剪枝后节点总数：", node_num)
        print("剪枝后叶子数：", len(res))
        print("规则数量：", len(res))
        # 显示前1条规则
        for i in range(1):
            print("R" + str(i + 1) + ":" + res[i].replace("AND -2.0 ", ""))  # 过滤最后叶子节点的值
        # 把规则存入文件
        file_write_obj = open("rules/rule2_post_prune.txt", 'w')
        for i in range(len(res)):
            file_write_obj.write(res[i])
            file_write_obj.write('\n')
        file_write_obj.close()  # 打开记得需要关闭

        return Best_tree, best_alpha, pruned_precision,train_pruned_precision, precision_list[0], train_score_list[0]


def CCP_top(prune=True, b_SE=True):
    clf, json_model = model_json()
    print("sklearn model has been transformed to json-style model=")
    if prune == False:
        return json_model, 0, 'you can calculate it with sklearn'
    else:
        alpha_list = []
        Ti_list = []
        print("unpruned model=", json_model)
        print("We are trying to get the Tree Sets,wait please..........")
        alpha_list, Ti_list = CCP_TreeCandidate(copy.deepcopy(clf), copy.deepcopy(json_model), feature_names,
                                                class_names, alpha_list, Ti_list)
        print("We have gotten all the Tree Sets.Validation is coming, please wait...............")
        print("alpha_list=", alpha_list)
        Best_tree, best_alpha, pruned_precision, train_pruned_precision, unpruned_precision,\
               train_unpruned_precision = CCP_validation(Ti_list,
                                                         alpha_list,
                                                         X_test,
                                                         y_test,
                                                         feature_names,
                                                         class_names,
                                                         copy.deepcopy(clf),
                                                         b_SE)
        # print("Best_tree=", Best_tree)
        print("best_alpha=", best_alpha)
        print("train_pruned_precision=", train_pruned_precision)
        print("pruned_precision=", pruned_precision)

        print("train_unpruned_precision=", train_unpruned_precision)
        print("unpruned_precision=", unpruned_precision)
        return Best_tree, best_alpha, pruned_precision, unpruned_precision


def CCP():
    Best_tree, best_alpha, pruned_precision, unpruned_precision = CCP_top(prune=True, b_SE=True)


if __name__ == '__main__':
    CCP()