from graphviz import Digraph
import pandas as pd
import math


def calcShannonEnt(dataSet, result_label):
    '''
    计算当前数据集的信息熵
    参数:
    dataSet:当前数据集（DataFrame)
    result:结果标签(string)
    返回值：
    ShannonEnt:信息熵(double)
    '''
    numEntries = len(dataSet)
    # 利用结果标签得到结果不同值的数量
    count = dataSet.groupby([result_label]).size()
    ShannonEnt = 0.0
    for attribute in count.keys():
        pro = count[attribute] / numEntries
        ShannonEnt = ShannonEnt - pro * math.log(pro, 2)
    return ShannonEnt


def chooseBestFeatureToSplit(dataSet, continuous_label, result_label):
    '''
    选择当前数据集中最好的划分标签和划分点
    参数:
    dataSet：当前数据集(DataFrame)
    continuous_label：连续的标签(list)
    result_lable：结果标签(string)
    返回值：
    bestFeature:最好的划分标签(string)
    bestpoint:最好的划分点(double)
    '''
    length = len(dataSet)
    baseEntropy = calcShannonEnt(dataSet, result_label)
    bestFeature = ''  # 最好的标签
    bestInfoGain = -1  # 最大信息增益
    bestpoint = float("inf")  # 最好的划分点
    # 得到当前数据集所拥有的标签
    attribute_list = list(dataSet.columns)[:-1]
    # 对于寻找划分标签，要分离散值和连续值
    for attribute in attribute_list:
        newEntropy = 0.0
        if attribute in continuous_label:
            # 连续值处理：排序，从排序好的数据中找到最好的划分点（遍历整个序列）和最好的划分标签
            continuous_list = sorted(list(dataSet[attribute]))
            for index in range(len(continuous_list) - 1):
                middle_value = (continuous_list[index] + continuous_list[index + 1]) / 2
                # 由中点划分成两个数据集
                subDatabase_1 = dataSet[dataSet[attribute] > middle_value]
                subDatabase_2 = dataSet[dataSet[attribute] <= middle_value]
                # 这里要注意精度，用newEnropy+=会造成错误
                newEntropy = (len(subDatabase_1) / length) * calcShannonEnt(subDatabase_1, result_label) + \
                             (len(subDatabase_2) / length) * calcShannonEnt(subDatabase_2, result_label)
                # 计算信息增益
                infoGain = baseEntropy - newEntropy
                if (infoGain > bestInfoGain):
                    bestInfoGain = infoGain
                    bestFeature = attribute
                    bestpoint = middle_value
        else:
            # 离散值处理：根据不同取值划分为不同的子集，再求子集的熵
            group = dataSet.groupby(attribute)
            for Set in group:
                subDatabase = Set[1]
                pro = len(subDatabase) / length
                newEntropy = newEntropy + pro * calcShannonEnt(subDatabase, result_label)
            # 计算信息增益
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestFeature = attribute
                bestInfoGain = infoGain
    return bestFeature, bestpoint


def majorityCnt(dataSet, result_label):
    '''
    返回当前数据集中根据结果标签划分之后数量最多的值（特征）
    参数：
    dataSet:当前数据集(DataFrame)
    result_label：结果标签(string)
    返回值：
    bestFeature:数量最多的值（特征）(string)
    用于剪枝和当剩余属性只有结果标签时
    '''
    size = dataSet.groupby([result_label]).size()
    bestFeature = ''
    maxsize = -1
    for attribute in size.keys():
        if (maxsize < size[attribute]):
            maxsize = size[attribute]
            bestFeature = attribute
    return bestFeature


def Get_Best_Pro(dataSet, result_label):
    '''
    得到当前数据集根据结果划分后所占比例最高值的比例
    参数：
    dataSet:当前数据集(DataFrame)
    result_label：结果标签(string)
    返回值：
    max_pro:最高比例(double)
    '''
    length = len(dataSet)
    max_pro = -1
    size = dataSet.groupby([result_label]).size()
    for attribute in size.keys():
        pro = size[attribute] / length
        if max_pro < pro:
            max_pro = pro
    return max_pro


def Purn(path, bestFeature, bestpoint, test_data, result_label, continuous_label):
    '''
    剪枝
    参数：
    path:决策树到当前节点之前的划分路径(list: [{标签:值},{标签:值}...])
    bestFeature:最好划分标签(string)
    bestpoint：最好划分点(double)
    test_data：测试集
    result_label:结果标签
    continuous_label：连续标签
    返回值
    True:可以剪枝
    False:不可以剪枝
    '''
    # 思路
    # 1. 用path划分test_data，在其剩下子数据集，根据结果标签划分后最大的概率即为当前节点的正确率(pre_pro)
    # 2. 在剩下的剩下子数据集中根据bestFeature划分，同样求出对应不同子集的最大概率，然后求加权和
    # 若1 > 2，证明划分后的正确率小于当前的正确率，则不划分，return majorityCnt(dataSet),若1 < 2,则继续划分
    subDatabase = test_data
    # 根据path划分得到对应的子数据集，注意要分连续和离散
    for attribute in path:
        for key, value in attribute.items():
            if key in continuous_label:
                if '>' in value:
                    mid_value = value[1:]
                    subDatabase = subDatabase[subDatabase[key] > float(mid_value)]
                else:
                    mid_value = value[2:]
                    subDatabase = subDatabase[subDatabase[key] <= float(mid_value)]
            else:
                subDatabase = subDatabase[subDatabase[key] == value]
        if (len(subDatabase) == 0):
            break
    if (len(subDatabase) == 0):
        return False
    pre_pro = 0
    pre_pro = Get_Best_Pro(subDatabase, result_label)
    post_pro = 0
    # 子数据集根据bestFeature划分
    if bestFeature in continuous_label:
        subDatabase_1 = subDatabase[subDatabase[bestFeature] > bestpoint]
        if len(subDatabase_1) != 0:
            post_pro += Get_Best_Pro(subDatabase_1, result_label)
        subDatabase_2 = subDatabase[subDatabase[bestFeature] <= bestpoint]
        if len(subDatabase_2) != 0:
            post_pro += Get_Best_Pro(subDatabase_2, result_label)
    else:
        group = subDatabase.groupby(bestFeature)
        length = len(subDatabase)
        for Set in group:
            subDatabase_1 = Set[1]
            pro = len(subDatabase_1) / length
            post_pro = post_pro + pro * Get_Best_Pro(subDatabase_1, result_label)
        # 比较划分前和划分后的正确率
    if pre_pro >= post_pro:
        return False
    else:
        return True


def createTree(dataSet, continuous_label, result_label, test_data, pre=0, post=0, path=[]):
    '''
    参数：
    dataSet:当前数据集(DataFrame)
    continuous_label：连续标签（list)
    result_label:结果标签(string)
    test_data：测试数据集
    pre:是否预剪枝(1,0)
    post:是否后剪枝(1,0)
    path:记录当前划分路径
    返回值：
    myTree（dic）
    '''
    attribute_list = list(dataSet.columns)[:-1]
    # 若dataSet只有结果属性，则输出结果属性中占比最多的值
    if (len(attribute_list) == 0):
        return majorityCnt(dataSet, result_label)
    # 若dataSet对于结果属性的分类后的某一个值和长度相等，则直接返回该值
    size = dataSet.groupby([result_label]).size()
    for attribute in size.keys():
        if (size[attribute] == len(dataSet)):
            return attribute
    # 根据算法选择分类的属性和节点
    bestFeature, bestpoint = chooseBestFeatureToSplit(dataSet, continuous_label, result_label)
    # 判断是否进行预剪枝
    if pre == 1:
        if not Purn(path, bestFeature, bestpoint, test_data, result_label, continuous_label):
            return majorityCnt(dataSet, result_label)
    # 构造字典树
    MyTree = {bestFeature: {}}
    if bestFeature in continuous_label:
        # 对于连续标签，划分之后无需将其删除，可以重复的使用
        bestpoint = round(bestpoint, 4)
        subDatabase_1 = dataSet[dataSet[bestFeature] > bestpoint]
        if len(subDatabase_1) != 0:
            path.append({bestFeature: '>' + str(bestpoint)})
            MyTree[bestFeature]['>' + str(bestpoint)] = createTree(subDatabase_1, continuous_label, result_label \
                                                                   , test_data, pre, post, path)
            path.pop()
        subDatabase_2 = dataSet[dataSet[bestFeature] <= bestpoint]
        if len(subDatabase_2) != 0:
            path.append({bestFeature: '<=' + str(bestpoint)})
            MyTree[bestFeature]['<=' + str(bestpoint)] = createTree(subDatabase_2, continuous_label, result_label, \
                                                                    test_data, pre, post, path)
            path.pop()
    else:
        group = dataSet.groupby(bestFeature)
        for Set in group:
            newDataSet = dataSet[dataSet[bestFeature] == Set[0]]
            path.append({bestFeature: Set[0]})
            newDataSet = newDataSet.drop(columns=bestFeature)  # 删除相应的属性的列，得到子数据集
            MyTree[bestFeature][Set[0]] = createTree(newDataSet, continuous_label, result_label, test_data, pre, post,
                                                     path)
            path.pop()
    # 判断是后剪枝
    if post == 1:
        if not Purn(path, bestFeature, bestpoint, test_data, result_label, continuous_label):
            return majorityCnt(dataSet, result_label)
    return MyTree


def Accuracy(Dic_Tree, test_data, continuous_label, result_label):
    '''得到当前决策树的准确值
    参数：
    Dic_Tree:字典树
    test_data:测试集(DataFrame)
    continuous_label:连续标签(list)
    result_label:结果标签
    返回值：
    accuracy:正确率(double)
    '''
    # default = test_data.iloc[0][result_label]
    length = len(test_data)
    count = 0
    for i in range(length):
        changed = False
        Temp_Dic = Dic_Tree
        temp_test_data = test_data.iloc[i]
        while True:
            if not isinstance(Temp_Dic, dict):
                break
            attribute = list(Temp_Dic.keys())[0]
            Temp_Dic = Temp_Dic[attribute]
            if attribute in continuous_label:
                key = list(Temp_Dic.keys())[0]
                if '>' in key:
                    mid_value = key[1:]
                else:
                    mid_value = key[2:]
                if temp_test_data[attribute] > float(mid_value):
                    Temp_Dic = Temp_Dic['>' + mid_value]
                else:
                    Temp_Dic = Temp_Dic['<=' + mid_value]
            else:
                flag = True
                for key in Temp_Dic.keys():
                    if temp_test_data[attribute] == key:
                        Temp_Dic = Temp_Dic[key]
                        flag = False
                if flag:
                    changed = True
                    break
        if temp_test_data[result_label] == Temp_Dic and not changed:
            count += 1
    # if changed:
    # 	if temp_test_data[result_label] == default:
    # 		count += 1
    return count / length


def Draw_Tree(Temp_Dic):
    '''
    绘制决策树
    参数：Temp_Dic：字典树
    思想：利用bsf进行遍历
    '''
    dot = Digraph(comment="Decision_Tree")
    lis = []
    lis.append(Temp_Dic)
    label_id = 0  # 使相同的内容表示为不同的结点
    while len(lis) != 0:
        temp = lis.pop(0)
        for pre in temp.keys():
            dot.node(name=pre, fontname='utf-8', shape='rect')
            subtemp = temp[pre]
            for value in subtemp.keys():
                subtemp_2 = subtemp[value]
                if not isinstance(subtemp_2, dict):
                    dot.node(subtemp_2 + str(label_id), subtemp_2, fontname='utf-8')
                    dot.edge(pre, subtemp_2 + str(label_id), label=value, fontname='utf-8')
                    label_id += 1
                else:
                    for after in subtemp_2.keys():
                        dot.edge(pre, after, label=value, fontname='utf-8')
                    lis.append(subtemp_2)

    import os
    os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'
    dot.view()


def prune(pre=0, post=0):
    training_data = pd.read_csv("watermelon3_0_Ch.csv").drop(columns="?编号")
    attributes = list(training_data.columns)
    test_data = []
    result_label = attributes[-1]
    continuous_label = attributes[-3:-1]  # 得到连续的元素
    tree = createTree(training_data, continuous_label, result_label, test_data, pre, post)
    Draw_Tree(tree)


if __name__ == '__main__':
    prune()