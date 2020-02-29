from pymongo import MongoClient
from matplotlib import pyplot as plt
from matplotlib import style
from exceptions import CollectionError
import time

# action = ["still", "turn_over", "legs_stretch", "hands_stretch",
#               "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

# 导入全局变量
import GlobalVariable as gv
action = gv.action_names


config = {'action': action,
          'db': 'beaglebone',
          'tag_collection': 'tags_1105',
          'volt_collection': 'volts_1105',
          'ndevices': 5,
          'offset': 0
          }

ndevices = 5
start = 5
end = start

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

    ntags = tag_collection.count_documents({'tag': action})
    # ntags = 12
    tag_acc = 0

    title = config['volt_collection'][6:] + "" + action
    fig = plt.figure(title, figsize=(6, 8))
    fig.suptitle(action)

    # plot the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        tag_acc += 1
        if (tag_acc > ntags):
            break
        # if (tag_acc == 9 or tag_acc == 11):  # don't discard data
        if (tag_acc == 9 or (tag_acc == 11 and action != "hands_move")):
            continue
        # inittime
        inittime, termtime = tag['inittime'] - offset, tag['termtime'] - offset
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
        subtitle = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        subtitle= ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

        ax = fig.add_subplot(ntags,1,tag_acc)
        plt.subplots_adjust(hspace=0.5)  # 函数中的wspace是子图之间的垂直间距，hspace是子图的上下间距
        ax.set_title("Person" + subtitle[tag_acc - 1] + ": " + timeToFormat(inittime + offset) + " ~ " + timeToFormat(
            termtime + offset))
        ax.set_xlim(inittime, termtime)

        # 自定义y轴的区间范围，可以使图放大或者缩小
        # ax.set_ylim([0.8,1.8])
        ax.set_ylim([0.75, 0.90])
        # ax.set_ylim([0.82, 0.83])
        ax.set_ylabel('Voltage(mv)')

        for i in range(start, end + 1):
            # [v + i*0.2 for v in volts[i]]为了把多个设备的数据隔离开
            ax.plot(times[i], volts[i], label='device_' + str(i), color=colors[i - 1], alpha=0.9)

        if tag_acc == 1:
            ax.legend(loc='upper right')
        if tag_acc == ntags:
            ax.set_xlabel('Time(mm:ss)')

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

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(20, 10)
    plt.savefig("action_images/" + title + str(start) + ".png", dpi=200)
    plt.close()

    # 最大化显示图像窗口
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.show()


if __name__ == '__main__':
    for i in range(len(action)):
        print("---------" + action[i] + "---------")
        plot_from_db(action=action[i],
                      db=config['db'],
                      tag_collection=config['tag_collection'],
                      volt_collection=config['volt_collection'],
                      ndevices=5,
                      offset=config['offset'])
