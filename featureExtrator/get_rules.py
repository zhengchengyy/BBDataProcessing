from sklearn.tree import _tree
import numpy as np


def get_paths(root, path, res, feature_name, label):
    feature_names = ["Range", "StandardDeviation"]
    if tree_.feature[root] != _tree.TREE_UNDEFINED:
        if(label==""):
            path.append(feature_name[root] + str(round(tree_.threshold[root], 3)))
        if(label=="left"):
            path.append("left"+feature_name[root]+str(round(tree_.threshold[root], 3)))
        if(label=="right"):
            path.append("right" + feature_name[root] + str(round(tree_.threshold[root], 3)))
        left = get_paths(tree_.children_left[root], path, res, feature_name, "left")
        right = get_paths(tree_.children_right[root], path, res, feature_name, "right")
        if not left and not right:  # 如果root是叶子结点
            res.append("->".join(path))  # 把当前路径加入到结果列表中
        path.pop()  # 返回上一层递归时，要让当前路径恢复原样
        return True
    # else:
    #     idx = np.argmax(tree_.value[root])
    #     # path.append(action_names[idx])
    #     # print("{}".format(action_names[idx]))


feature_names = ["Range", "StandardDeviation"]
action_names = ["turn_over", "legs_stretch", "hands_stretch",
                "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

# 读取模型
import pickle
with open('models/0.966model.pickle', 'rb') as f:
    model = pickle.load(f)

tree_ = model.tree_
feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
print(tree_.n_leaves)


res = []
get_paths(0, [], res, feature_name, "")
print(len(res))
print(res)