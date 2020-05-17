from pymongo import MongoClient
from matplotlib import pyplot as plt
from matplotlib import style
from exceptions import CollectionError
import time

action = ["turn_over","legs_stretch","hands_stretch",
          "legs_tremble","hands_tremble","body_tremble",
          "head_move","legs_move","hands_move",
          "hands_rising","kick"]
# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names

config = {'action': action[1],
          'db': 'beaglebone',
          'tag_collection': 'tags_1105',
          'volt_collection': 'volts_1105',
          'ndevices': 5,
          'offset': 0
          }

ndevices = config['ndevices']
# 统计不同设备动作原始数据的个数
origin_num = {}
for i in range(1, ndevices + 1):
    origin_num[i] = []

def timeToFormat(t):
    ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return ftime


def timeToSecond(t):
    stime = time.strftime("%M:%S", time.localtime(t))
    return stime


def plot_from_db(action, db, volt_collection, tag_collection, port=27017, host='localhost', ndevices=3, offset=0):
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found, please check names of the collection and database')
    except CollectionError as e:
        print(e.message)

    # ntags表示总标签数，即人数；tag_acc表示累加计数
    ntags = tag_collection.count_documents({'tag': action})
    # ntags = 1
    tag_acc = 0

    # 查看第几号设备
    start = 1
    end = ndevices

    # 存储一个动作所有人数的列表
    one_action_all_people = {}
    for i in range(1, ndevices + 1):
        one_action_all_people[i] = []

    # plot the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        tag_acc += 1
        if (tag_acc > ntags):
            break
        # if (tag_acc == 9 or tag_acc == 11):  # don't discard data

        if (tag_acc == 9 or (tag_acc == 11 and action != "hands_move")):
            continue
        inittime, termtime = tag['inittime'], tag['termtime']
        times, volts = {}, {}
        for i in range(1, ndevices + 1):
            times[i] = []
            volts[i] = []

        for volt in volt_collection.find({'time': {'$gt': inittime, '$lt': termtime}}):
            device_no = int(volt['device_no'])
            v = volt['voltage']
            t = volt['time']
            times[device_no].append(t)
            volts[device_no].append(v)

        for i in range(1, ndevices + 1):
            one_action_all_people[i].append(len(volts[i]))
            # print(len(volts[i]))

    for i in range(1, ndevices + 1):
        origin_num[i].append(one_action_all_people[i])




if __name__ == '__main__':
    for i in range(len(action_names)):
        plot_from_db(action=action_names[i],
                     db=config['db'],
                     tag_collection=config['tag_collection'],
                     volt_collection=config['volt_collection'],
                     ndevices=config['ndevices'],
                     offset=config['offset'])

    for i in range(1, ndevices + 1):
        print(origin_num[i])
        # print(sum(origin_num[i]))

    # 保存为np文件，三维列表[设备号，动作号，人员号）
    import numpy as np
    temp = np.array(origin_num)
    np.save('origin_num.npy', temp)
