from pymongo import MongoClient
from matplotlib import pyplot as plt
from matplotlib import style
from exceptions import CollectionError
import time
import numpy as np
import pywt
import math

action = ["still", "turn_over", "legs_stretch", "hands_stretch",
          "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

config = {'action': action[1],
          'db': 'beaglebone',
          'tag_collection': 'tags_411',
          'volt_collection': 'volts_411',
          'ndevices': 3,
          'offset': 0
          }

action_names = [
          "get_up","go_to_bed",
          "turn_over","legs_stretch","hands_stretch",
          "head_move","legs_move","hands_move",
          "kick","legs_tremble","hands_tremble"]

config = {'action': "turn_over",
          'db': 'beaglebone',
          'tag_collection': 'tags_1105',
          'volt_collection': 'volts_1105',
          'ndevices': 5,
          'offset': 0
          }


def timeToFormat(t):
    ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return ftime


def timeToSecond(t):
    stime = time.strftime("%M:%S", time.localtime(t))
    return stime


def cwt_filter(data, threshold):
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    data_filter = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    if (len(data) != len(data_filter)):
        data_filter = np.delete(data_filter, 0)

    return data_filter


# 计算均方误差
def get_rmse(volts, volts_filter):
    if len(volts) == len(volts_filter):
        mse = sum([(x - y) ** 2 for x, y in zip(volts, volts_filter)]) / len(volts)
        return math.sqrt(mse)
    else:
        return None


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
    ntags = 1
    tag_acc = 0

    # 查看第几号设备
    start = 1
    end = 1

    title = config['volt_collection'][6:] + "" + action + "_cwt_filter"
    fig = plt.figure(title)
    fig.suptitle(action + "_cwt_filter")

    # plot the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        tag_acc += 1
        if (tag_acc > ntags):
            break
        # inittime,termtime
        inittime, termtime = tag['termtime'] - offset - 31, tag['termtime'] - offset
        # get the arrays according to which we will plot later
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

        style.use('default')
        colors = ['r', 'b', 'g', 'c', 'm']  # m c
        subtitle = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N']

        ax = fig.add_subplot(ntags, 1, tag_acc)

        plt.subplots_adjust(hspace=0.5)  # 函数中的wspace是子图之间的垂直间距，hspace是子图的上下间距
        ax.set_title("Person" + subtitle[tag_acc - 1] + ": " + timeToFormat(inittime + offset) + " ~ " + timeToFormat(
            termtime + offset))
        ax.set_xlim(inittime, termtime)

        # 自定义y轴的区间范围，可以使图放大或者缩小
        # ax.set_ylim([0.8,1.8])
        ax.set_ylim([0.75, 0.90])
        ax.set_ylim([0, 1])
        # ax.set_ylim([0.82, 0.83])
        ax.set_ylabel('Voltage(mv)')

        for i in range(start, end + 1):
            volts_filter = volts[i]
            # filter_thread = [0.2, 0.06, 0.08]
            # volts_filter = cwt_filter(volts[i], filter_thread[i-1])  #411数据专用
            volts_filter = cwt_filter(volts[i], 0.1)
            print("device"+str(i)+"'RMSE:",get_rmse(volts[i], volts_filter))
            ax.plot(times[i], volts_filter, label='device_' + str(i), color=colors[i - 1], alpha=0.9)

        ax.grid(linestyle=':')
        if tag_acc == 1:
            # ax.legend(loc='upper right')
            pass
        if tag_acc == ntags:
            ax.set_xlabel('Time(s)')

        # 以第一个设备的时间数据为准，数据的每1/10添加一个x轴标签
        xticks = []
        xticklabels = []
        length = len(times[1])
        interval = length // 15 - 1
        for i in range(0, length, interval):
            xticks.append(times[1][i])
            # xticklabels.append(timeToSecond(times[1][i] + offset))
            xticklabels.append(int(times[1][i] - inittime))  # 图中的开始时间表示时间间隔interval
        ax.set_xticks(xticks)  # 设定标签的实际数字，数据类型必须和原数据一致
        ax.set_xticklabels(xticklabels, rotation=15)  # 设定我们希望它显示的结果，xticks和xticklabels的元素一一对应

        # 最大化显示图像窗口
        # plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


if __name__ == '__main__':
    plot_from_db(action=config['action'],
                 db=config['db'],
                 tag_collection=config['tag_collection'],
                 volt_collection=config['volt_collection'],
                 ndevices=config['ndevices'],
                 offset=config['offset'])
