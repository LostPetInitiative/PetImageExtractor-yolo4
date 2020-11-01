import json
import base64
import tempfile
import os
from skimage import io
import numpy as np
import shutil
import kafka
from kafka.admin import KafkaAdminClient, NewTopic

def strSerializer(jobName):
    return jobName.encode('utf-8')

def strDeserializer(jobNameBytes):
    return jobNameBytes.decode('utf-8')

def dictSerializer(job):
    #print(type(job))
    #print(job)
    return json.dumps(job, indent=2).encode('utf-8')

def dictDeserializer(jobBytes):
    #print(type(job))
    #print(job)
    return json.loads(jobBytes.decode('utf-8'))

def imagesNpToStrList(npImages):
    tempDir = tempfile.mkdtemp()
    try:
        idx1 = 0
        images = []
        # encoding images
        for npImage in npImages:
            photoPath = os.path.join(tempDir,"{0}.jpeg".format(idx1))
            io.imsave(photoPath, npImage)
            #print("image {0} saved".format(photoPath))
            with open(photoPath, 'rb') as photoFile:
                photo = photoFile.read()
                #print("image {0} read".format(photoPath))
                image = {
                    'type': "jpg",
                    'data': base64.encodebytes(photo).decode("utf-8").replace("\n","")
                }
                images.append(image)
            idx1 += 1
        return images
    finally:
        shutil.rmtree(tempDir)

def imagesFieldToNp(images):
    tempDir = tempfile.mkdtemp()
    try:
        imgIdx = 0
        imagesNp = []
        # decoding images
        for image in images:
            imgType = image['type']
            image_b64 : str = image['data']
            imageData = base64.decodebytes(image_b64.encode("utf-8"))
            imageFilePath = os.path.join(tempDir,"{0}.{1}".format(imgIdx,imgType))
            with open(imageFilePath, "wb") as file1:             
                file1.write(imageData)
            try:
                if imageFilePath.endswith(".png"):
                    imNumpy = io.imread(imageFilePath,plugin='imread')
                else:
                    imNumpy = io.imread(imageFilePath)
                imagesNp.append(imNumpy)
            except Exception as exc1:
                print("Error calulating hash for one of the images ({0})".format(exc1))        
            imgIdx += 1
        return imagesNp
    finally:
        shutil.rmtree(tempDir)
                


class JobQueue:
    def __init__(self, kafkaBootstrapUrl,topicName, appName, num_partitions=8, replication_factor=3, retentionHours = 7*24):
        self.kafkaBootstrapUrl = kafkaBootstrapUrl
        self.topicName = topicName
        self.appName = appName
        admin_client = KafkaAdminClient(
            bootstrap_servers=kafkaBootstrapUrl, 
            client_id=appName
            )

        topic_list = []
        topic_configs = {
            #'log.retention.hours': str(retentionHours)
        }
        topic_list.append(NewTopic(name=topicName, num_partitions=num_partitions, replication_factor=replication_factor,topic_configs=topic_configs))
        topics = admin_client.list_topics()
        if not (topicName in topics):
            try:
                admin_client.create_topics(new_topics=topic_list, validate_only=False)
                print("Topic {0} is created".format(topicName))
            except kafka.errors.TopicAlreadyExistsError:
                print("Topic {0} already exists".format(topicName))
        else:
            print("Topic {0} already exists".format(topicName))
        admin_client.close()


class JobQueueProducer(JobQueue):
    '''Posts Jobs as JSON serialized python dicts'''
    def __init__(self, *args, **kwargs):
        super(JobQueueProducer, self).__init__(*args, **kwargs)

        self.producer = kafka.KafkaProducer( \
            bootstrap_servers = self.kafkaBootstrapUrl, \
            client_id = self.appName,
            key_serializer = strSerializer,
            value_serializer = dictSerializer,
            compression_type = "gzip")

    async def Enqueue(self, jobName, jobBody):
        return self.producer.send(self.topicName, value=jobBody, key= jobName)

class JobQueueWorker(JobQueue):
    '''Fetchs sobs as JSON serialized python dicts'''
    def __init__(self, group_id, *args, **kwargs):
        super(JobQueueWorker, self).__init__(*args, **kwargs)

        self.teardown = False
        self.consumer = kafka.KafkaConsumer(self.topicName, \
            bootstrap_servers = self.kafkaBootstrapUrl, \
            client_id = self.appName,
            group_id = group_id,
            key_deserializer = strDeserializer,
            value_deserializer = dictDeserializer)

    def GetNextJob(self, pollingIntervalMs = 1000):
        extracted = False
        while (not self.teardown) and (not extracted):
            res = self.consumer.poll(pollingIntervalMs, max_records=1)
            #print("Got {0}. Len {1}".format(res,len(res)))
            if(len(res) == 1):
                for key in res:
                    jobValue = res.get(key)[0].value
                    return jobValue        

    def TryGetNextJob(self, pollingIntervalMs = 1000):
        res = self.consumer.poll(pollingIntervalMs, max_records=1)
        #print("Got {0}. Len {1}".format(res,len(res)))
        if(len(res) == 1):
            for key in res:
                jobValue = res.get(key)[0].value
                return jobValue
        else:
            return None


    def Commit(self):
        self.consumer.commit()

    def Close(self, autocommit=True):
        self.Close(autocommit)