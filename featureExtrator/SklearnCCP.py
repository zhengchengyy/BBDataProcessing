from SklearnToJson import model_json, draw_file
import pandas as pd
import numpy as np
import re
import copy
from sklearn.tree._tree import TREE_LEAF
from math import sqrt


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
    png_file = "visualization/T" + Tt_name + ".svg"
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
    print("current_json_model=", current_json_model)

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
    progress_length = len(TreeSets)

    for index, item in enumerate(TreeSets):
        Ti_precision = precision_compute(item, X_test, y_test, feature_names, class_names)
        print("T%d_precision=%f" % (index, Ti_precision))
        precision_list.append(Ti_precision)
        print("the T" + str(index) + " has been validated, " + str(
            progress_length - index - 1) + " Trees left, please wait.....")
    if b_SE == False:
        pruned_precision = max(precision_list)
        index = precision_list.index(pruned_precision)
        print("index=", index)
        best_alpha = alpha_list[index]
        Best_tree = TreeSets[index]
        dot_file = "visualization/Best_tree_0SE.dot"
        svg_file = "visualization/Best_tree_0SE.svg"
        # 画一画树

        best_sklearn_model = copy.deepcopy(sklearn_model)
        prune_sklearn_model(best_sklearn_model.tree_, 0, Best_tree)

        draw_file(best_sklearn_model, dot_file, svg_file, feature_names)
        return Best_tree, best_alpha, pruned_precision, precision_list[0]

    else:  # 使用1-SE rule
        error_rate_list = [1 - item for item in precision_list]
        lowest_error_rate = min(error_rate_list)
        print("error_rate_list=", error_rate_list)
        SE = sqrt(lowest_error_rate * (1 - lowest_error_rate) / len(y_test))
        print("SE=", SE)

        criterion_1_SE = lowest_error_rate + SE

        index_error_rate = 0
        for index, item in enumerate(
                error_rate_list):  # search from from the end ,because the error_rate_list is not monotory.

            if error_rate_list[len(error_rate_list) - 1 - index] < criterion_1_SE:
                index_error_rate = len(error_rate_list) - 1 - index
                break

        # if index_error_rate-1>=0:
        #     index_error_rate=index_error_rate-1
        # else:
        #     pass#becasuse the list may only have one item.

        # here's right,because the precision list is corresponding to the error_rate_list.
        pruned_precision = precision_list[index_error_rate]

        best_alpha = alpha_list[index_error_rate]
        Best_tree = TreeSets[index_error_rate]
        dot_file = "visualization/Best_tree_1SE.dot"
        svg_file = "visualization/Best_tree_1SE.svg"
        # 画一画树
        best_sklearn_model = copy.deepcopy(sklearn_model)
        prune_sklearn_model(best_sklearn_model.tree_, 0, Best_tree)

        draw_file(best_sklearn_model, dot_file, svg_file, feature_names, class_names)
        return Best_tree, best_alpha, pruned_precision, precision_list[0]


def CCP_top(prune=True, b_SE=True):
    clf, json_model, X_train, y_train, X_test, y_test, feature_list, class_names = model_json()
    print("sklearn model has been transformed to json-style model=\n")
    if prune == False:
        return json_model, 0, 'you can calculate it with sklearn'
    else:
        alpha_list = []
        Ti_list = []
        print("unpruned model=", json_model)
        print("We are trying to get the Tree Sets,wait please..........")
        alpha_list, Ti_list = CCP_TreeCandidate(copy.deepcopy(clf), copy.deepcopy(json_model), feature_list,
                                                class_names, alpha_list, Ti_list)
        print("We have gotten all the Tree Sets.Validation is coming, please wait...............")
        print("alpha_list=", alpha_list)
        Best_tree, best_alpha, pruned_precision, unpruned_precision = CCP_validation(Ti_list,
                                                                                     alpha_list,
                                                                                     X_test,
                                                                                     y_test,
                                                                                     feature_list,
                                                                                     class_names,
                                                                                     copy.deepcopy(clf),
                                                                                     b_SE)
        print("Best_tree=", Best_tree)
        print("best_alpha=", best_alpha)
        print("pruned_precision=", pruned_precision)
        return Best_tree, best_alpha, pruned_precision, unpruned_precision


def CCP():
    Best_tree, best_alpha, pruned_precision, unpruned_precision = CCP_top(prune=True, b_SE=True)


if __name__ == '__main__':
    CCP()
