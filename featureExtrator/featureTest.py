from feature_extractor import SumModule
from feature_extractor import FeatureExtractor
from threading import Thread
import time


class NumberGenerator(Thread):
    """一个数字生成器，每隔1秒生成一个从0递增的数字。"""

    def __init__(self, featureExtractor):
        self.featureExtractor = featureExtractor
        super(NumberGenerator, self).__init__()

    def run(self):
        n = 0
        while True:
            n = n + 1
            self.featureExtractor.process(n)
            time.sleep(1)


summodule = SumModule(5, 2)
extractor = FeatureExtractor()
extractor.register(summodule)
mainThread = NumberGenerator(extractor)
mainThread.start()
