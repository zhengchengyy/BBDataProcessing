def getNormalization(li):
    temp = []
    _max = max(li)
    _min = min(li)
    print(_max, _min)
    for i in li:
        normal = (i - _min) / (_max - _min)
        temp.append(normal)
    return temp

li = [1,2,3,4,5,1,5,5]
weights = [65, 75]
li = list(map(lambda x: x / weights[0], li))
print(li)
result = getNormalization(li)
print(result)
# weights[0] [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 1.0, 1.0]
# weights[1]  [0.0, 0.25000000000000006, 0.5, 0.7500000000000001, 1.0, 0.0, 1.0, 1.0]