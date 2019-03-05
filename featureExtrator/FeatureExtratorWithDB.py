from FeatureModules import *
from feature_extractor import FeatureExtractor
import time

from pymongo import MongoClient
from exceptions import CollectionError


config = {'action':'uneasy',
          'db':'beaglebone',
          'tag_collection':'tags_5',
          'volt_collection':'volts_5'}

def read_from_db(action, db, volt_collection, tag_collection,port=27017, host='localhost', ndevices=3):
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found, please check names of the collection and database')
    except CollectionError as e:
        print(e.message)

    ntags = tag_collection.count_documents({'tag':action})

    # read the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        inittime, termtime = tag['inittime'], tag['termtime']

        # get the arrays according to which we will plot later
        times, volts = {}, {}
        for i in range(1, ndevices + 1):
            times[i] = []
            volts[i] = []

        for volt in volt_collection.find({'time': {'$gt': inittime,'$lt': termtime}}):
            device_no = int(volt['device_no'])
            v = volt['voltage']
            time = volt['time']
            times[device_no].append(time)
            volts[device_no].append(v)

    return times,volts


if __name__=='__main__':
    times, volts= read_from_db(action=config['action'],
                               db=config['db'],
                               tag_collection=config['tag_collection'],
                               volt_collection=config['volt_collection'])

    maxsize = 5
    leapsize = 3

    extractor = FeatureExtractor()

    variancemodule = VarianceModule(5, 3)
    averagemodule = AverageModule(5, 3)
    thresholdcounter = ThresholdCounterModule(5, 3)

    extractor.register(variancemodule)
    extractor.register(averagemodule)
    extractor.register(thresholdcounter)
    for volt in volts[1]:
        #print(volt)
        time.sleep(1)
        extractor.process(volt)
