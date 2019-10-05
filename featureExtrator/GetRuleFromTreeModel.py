from sklearn.tree import _tree
import numpy as np


def binaryTreePaths(root, feature_names, action_names):
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    if root == _tree.TREE_UNDEFINED:
        return []
    res, stack = [], [(root, "")]
    while stack:
        node, ls = stack.pop()
        left = tree_.children_left[node]
        right = tree_.children_right[node]
        val = str(round(tree_.threshold[node], 3))
        name = feature_name[node]
        if left == _tree.TREE_LEAF and right == _tree.TREE_LEAF:
            idx = np.argmax(tree_.value[node])
            result = " THEN " + action_names[idx]
            res.append("IF " + ls + val + result)
        if right != _tree.TREE_LEAF:
            stack.append((right, ls + name + " > " + val + " AND "))
        if left != _tree.TREE_LEAF:
            stack.append((left, ls + name + " <= " + val + " AND "))
    return res


# 读取模型
import pickle

with open('models/0.966model.pickle', 'rb') as f:
    model = pickle.load(f)

feature_names = ["Range", "StandardDeviation"]
action_names = ["turn_over", "legs_stretch", "hands_stretch",
                "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

tree_ = model.tree_
print("叶子数量：", tree_.n_leaves)

res = binaryTreePaths(0, feature_names, action_names)
print("规则数量：", len(res))

# print(res)
# 显示前10条规则
for i in range(10):
    print("R" + str(i + 1) + ":" + res[i].replace("AND -2.0 ", ""))  # 过滤最后叶子节点的值

# 把规则存入文件
file_write_obj = open("rules/rule2.txt", 'w')
for i in range(len(res)):
    file_write_obj.write("R" + str(i + 1) + ":" + res[i].replace("AND -2.0 ", ""))
    file_write_obj.write('\n')
