from feature_extractor import FeatureExtractor
from feature_extractor import ProcessModule


class ThresholdCounterModule(ProcessModule):
    """求超过指定阈值的个数"""

    FEATURE_NAME = "ThresholdCounter"
    UPPER_THRESHOLD = 0.80
    LOWER_THRESHOLD = 0.80

    def processFullQueue(self):
        count = 0
        for value in self.queue.queue:
            #仅在次数将value改为value['volt']
            if value['volt'] > self.UPPER_THRESHOLD or value['volt'] < self.LOWER_THRESHOLD:
                count = count + 1
        return count

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class VarianceModule(ProcessModule):
    """功能是对满队列中的所有数据求方差。返回方差。
    表示震动频率的平均情况"""

    FEATURE_NAME = "Variance"

    def processFullQueue(self):
        sum = 0
        variance = 0
        for value in self.queue.queue:
            sum = sum + value['volt']
        average = sum/self.size
        for value in self.queue.queue:
            variance = variance + (value['volt'] - average) ** 2
        return variance/self.size

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class AverageModule(ProcessModule):
    """功能是对满队列中的所有数据求平均值。返回平均值。
    表示震动幅度的平均情况"""
    # 优化，不用重复计算值，只计算增加和减少的数据；可以从其它组件获取数据

    FEATURE_NAME = "Average"

    def processFullQueue(self):
        sum = 0
        for value in self.queue.queue:
            sum = sum + value['volt']
        return sum/self.size

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()