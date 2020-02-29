from sklearn.tree import _tree
import numpy as np
import pickle

ndevices = 5
start = 1
end = ndevices


def binaryTreePaths(model, root, feature_names, action_names):
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature
    ]
    if root == _tree.TREE_UNDEFINED:
        return []
    rule_list, stack, proba_list = [], [(root, "")], []
    while stack:
        node, conditions = stack.pop()
        left = tree_.children_left[node]
        right = tree_.children_right[node]
        threshold = str(round(tree_.threshold[node], 3))
        name = feature_name[node]
        if left == _tree.TREE_LEAF and right == _tree.TREE_LEAF:
            idx = np.argmax(tree_.value[node])
            value = tree_.value[node]
            action_proba = value[0][idx] / sum(value[0])
            # consequent = " THEN action = " + action_names[idx] + ", action_proba = " + str(action_proba)
            consequent = " THEN action_num = " + str(idx) + ", action_proba = " + str(round(action_proba, 3))

            # rule_list.append("IF " + conditions + threshold + consequent)
            # 路径遍历得到的规则，按照动作概率从小到大的顺序，折半查找插入位置后插入规则列表中
            if (len(proba_list) == 0 or action_proba > proba_list[-1]):
                rule_list.append("IF " + conditions + threshold + consequent)
                proba_list.append(action_proba)
            else:
                low = 0
                high = len(proba_list) - 1
                while (low <= high):
                    mid = (low + high) >> 1
                    if (action_proba < proba_list[mid]):
                        high = mid - 1
                    else:
                        low = mid + 1
                rule_list.insert(low, "IF " + conditions + threshold + consequent)
                proba_list.insert(low, action_proba)

        if right != _tree.TREE_LEAF:
            stack.append((right, conditions + name + " > " + threshold + " AND "))
        if left != _tree.TREE_LEAF:
            stack.append((left, conditions + name + " <= " + threshold + " AND "))
    return rule_list


def extractRule(device_no):
    with open('models/' + 'device_' + str(device_no) + '_post_prune.pickle', 'rb') as f:
        model = pickle.load(f)

    # 导入全局变量
    import GlobalVariable as gv

    feature_names = gv.feature_names
    # 删除名字后缀
    feature_names = [feature[:-6] for feature in feature_names]
    action_names = gv.action_names

    tree_ = model.tree_
    print("节点总数：", tree_.node_count)
    print("叶子数量：", tree_.n_leaves)

    rule_list = binaryTreePaths(model, 0, feature_names, action_names)
    print("规则数量：", len(rule_list))

    # 显示前10条规则
    for i in range(1):
        # print("R" + str(i + 1) + ":" + rule_list[i])
        print("R" + str(i + 1) + ":" + rule_list[i].replace("AND -2.0 ", ""))  # 过滤最后叶子节点的值

    # 把规则存入文件
    file_write_obj = open("rules/rule" + '_device_' + str(device_no) + '.txt', 'w')
    for i in range(len(rule_list)):
        file_write_obj.write("R" + str(i + 1) + ":" + rule_list[i].replace("AND -2.0 ", ""))
        file_write_obj.write('\n')
        file_write_obj.write('\n')

    file_write_obj.close()


for i in range(start, end + 1):
    print("---------device_" + str(i) + " write rule file begin---------")
    extractRule(i)
    print("---------device_" + str(i) + " write rule file end---------")
