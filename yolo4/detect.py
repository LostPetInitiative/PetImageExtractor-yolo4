import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
from skimage import io
import numpy as np
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import gc


import kafkaJobQueue

import shutil
import os

kafkaUrl = os.environ['KAFKA_URL']
inputQueueName = os.environ['INPUT_QUEUE']
outputQueueName = os.environ['OUTPUT_QUEUE']

appName = "PetImageDetectionDetector"

worker = kafkaJobQueue.JobQueueWorker(appName, kafkaBootstrapUrl=kafkaUrl, topicName=inputQueueName, appName=appName)
resultQueue = kafkaJobQueue.JobQueueProducer(kafkaUrl, outputQueueName, appName)

cpuCount = multiprocessing.cpu_count()



flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416-coco',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
#flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
#flags.DEFINE_string('output', 'result.png', 'path to output image')
#flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

# cat class 15
# dog class 16

def log(message):
    print(message)

def main(_argv):
    async def work():
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = FLAGS.size

        log("Executing eagerly: {0}".format(tf.executing_eagerly()))

        log("Service started. Pooling for a job")
        infer = None
        while True:        
            job = worker.TryGetNextJob(5000)
            if job is None:
                if not(infer is None):
                    # unloading the model to free the memory
                    saved_model_loaded = None
                    infer = None
                    tf.keras.backend.clear_session()
                    gc.collect()
                    log("Model unloaded to free the memory")
                continue
            else:
                #print("Got job {0}".format(job))
                uid = job["UID"]
                log("{0}: Starting to process the job".format(uid))
                images = job['images']
                log("{0}: Extracting {1} images".format(uid, len(images)))
                
                imagesNp = kafkaJobQueue.imagesFieldToNp(images)

                if infer is None:
                    print("Loading model {0}".format(FLAGS.weights))
                    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
                    print("(Re)Loaded model {0}".format(FLAGS.weights))
                    infer = saved_model_loaded.signatures['serving_default']

                threadPool = ThreadPoolExecutor(max_workers=cpuCount*2)    
                #threadPool = ThreadPoolExecutor(max_workers=1)    
                gpuPool = ThreadPoolExecutor(max_workers=1)    
                loop = asyncio.get_running_loop()
                imageProcessingSemaphore = asyncio.Semaphore(cpuCount-1)
                
                def loadImage(npImage):
                    original_image = npImage
                    
                    imShape = original_image.shape
                    #print("imhsape is {0}".format(imShape))
                    if len(imShape) == 2: # greyscale
                        original_image = np.copy(np.tile(np.expand_dims(original_image, axis=2),(1,1,3)))
                    else:    
                        if imShape[2] == 4: #RGBA
                            original_image = np.copy(original_image[:,:,0:3])
                    #print("fixed hsape is {0}".format(original_image.shape))
                    
                    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
                    image_data = cv2.resize(original_image, (input_size, input_size))
                    image_data = image_data / 255.
                    # image_data = image_data[np.newaxis, ...].astype(np.float32)
                    return original_image, image_data

                def rotateImage(image_data, augNum):
                    rot = augNum & 1
                    horFlip = augNum & 2
                    vertFlip = augNum & 4

                    rotated = image_data
                    # if augNum  > 0:
                    #     #rotated = np.rot90(rotated, k=2)
                    #     rotated = np.fliplr(np.flipud(rotated))

                    if augNum % 4 > 0:
                        rotated2 = np.copy(np.rot90(rotated, k= augNum % 4))
                    else:
                        rotated2 = rotated
                    if augNum > 3:
                        rotated3 = np.copy(np.flipud(rotated2))
                    else:
                        rotated3 = rotated2
                    # if horFlip == 1:
                    #     rotated = np.fliplr(rotated)
                    # if vertFlip == 1:
                    #     rotated = np.flupud(rotated)
                    # if rot == 1:
                    #     rotated = np.rot90(rotated, k=1)
                    # if horFlip == 1:
                    #     rotated = np.rot90(rotated, k=1)
                    # if vertFlip == 1:
                    #     rotated = np.flupud(rotated)
                    # if rot == 1:
                    #     rotated = np.rot90(rotated, k=1)
                    return rotated3

                def rotateImageExt(image_data, rotation):
                    images_data = []
                    images_data.append(rotateImage(image_data, rotation))
                    images_data = np.asarray(images_data).astype(np.float32)
                    return images_data, rotation
                
                # def getBboxesStub(images_data, targetClass:int):
                #     return [np.zero, npScores, npClasses, npValidDetections]

                def getBboxes(images_data, targetClass:int):
                    #print("images_data shape {0}".format(images_data.shape))
                
                    batch_data = tf.constant(images_data)
                    #print("batch data {0}".format(type(batch_data)))
                    #exit(1)
                    pred_bbox = infer(batch_data) # works only for batch size 1 and 2 for some reason
                    for key, value in pred_bbox.items():
                        boxes = value[:, :, 0:4]
                        pred_conf = value[:, :, 4:]

                    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                            # A 4-D float Tensor of shape [batch_size, num_boxes, q, 4].
                            # If q is 1 then same boxes are used for all classes otherwise,
                            # if q is equal to number of classes, class-specific boxes are used. 
                        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                            # A 3-D float Tensor of shape [batch_size, num_boxes, num_classes]
                            # representing a single score corresponding to each box (each row of boxes). 
                        scores=tf.reshape(
                            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])), \
                            # the maximum number of boxes to be selected by non-max suppression per class 
                        max_output_size_per_class=50,
                        max_total_size=50,
                        iou_threshold=0.5,
                        score_threshold=FLAGS.score
                    )
                    # 'nmsed_boxes':    A [batch_size, max_detections, 4] float32 tensor containing
                    #                   the non-max suppressed boxes.
                    # 'nmsed_scores':   A [batch_size, max_detections] float32 tensor containing
                    #                   the scores for the boxes.
                    #  'nmsed_classes': A [batch_size, max_detections] float32 tensor containing
                    #                   the class for boxes.
                    #  'valid_detections':  A [batch_size] int32 tensor indicating
                    #                       the number of valid detections per batch item.
                    #                       Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
                    #                       The rest of the entries are zero paddings.
                    
                    # here we leave only target class

                    # boxes = tf.squeeze(boxes, axis=0)
                    # scores = tf.squeeze(scores, axis=0)
                    classesSqueezed = tf.squeeze(classes, axis=0)

                    validClassMask = tf.equal(classesSqueezed,targetClass)
                    validClassIndices = tf.squeeze(tf.where(validClassMask),axis=1)
                    validClassCount = tf.shape(validClassIndices)

                    boxes = tf.gather(boxes, validClassIndices, axis=1)
                    scores = tf.gather(scores, validClassIndices, axis=1)
                    classes = tf.gather(classes, validClassIndices, axis=1)

                    #print("valid class indices {0}".format(validClassIndices))

                    npBoxes = boxes.numpy()
                    npScores = scores.numpy()
                    npClasses = classes.numpy()
                    npValidDetections = validClassCount.numpy()

                    pred_bbox = [npBoxes, npScores, npClasses, npValidDetections]
                    return pred_bbox

                async def FindBestRotation(npImage, targetClass):
                    """Returns (annotatedImage,extractedPetImage, bestScore, bestRotation) or quadro-tuple on Nones if the pet is not detected"""
                    original_image, image_data = await loop.run_in_executor(threadPool, loadImage, npImage)
                    loadImageJobs = [loop.run_in_executor(threadPool,rotateImageExt, image_data, k) for k in range(8)]
                    bestScore = 0.0
                    bestScoreBboxes = None
                    noRotationValid = False
                    spoiled = False
                    bestRotation = 0
                    for coro in asyncio.as_completed(loadImageJobs):
                        images_data, rotation = await coro
                        bboxes = await loop.run_in_executor(gpuPool,getBboxes, images_data, targetClass)
                        if noRotationValid or spoiled:
                            continue
                        score = bboxes[1]
                        scoreShape = score.shape
                        detectedCount = scoreShape[1]
                        if (rotation == 0) and (detectedCount>1):
                            # detected multiple pets on the original image. discarding all series
                            spoiled = True
                            bestScoreBboxes = None
                        if detectedCount != 1:
                            continue # we are interested in the case when only single pet detected
                        if rotation == 0:
                            noRotationValid = True
                        scoreVal = score[0,0]
                        if (scoreVal > bestScore) or noRotationValid:
                            # selecting the rotation which gives the highest score
                            # absence of rotation has still higher priority
                            bestScore = scoreVal
                            bestScoreBboxes = bboxes
                            bestRotation = rotation
                    if not(bestScoreBboxes is None):
                        # found a rotation when a single pet of proper type is detected
                        if bestRotation != 0:
                            rotatedImage = rotateImage(original_image,bestRotation)
                        else:
                            rotatedImage = original_image

                        #rotatedImage = cv2.cvtColor(rotatedImage, cv2.COLOR_BGR2RGB)
                        annotatedImage = np.array(rotatedImage)
                        annotatedImage = utils.draw_bbox(annotatedImage, bestScoreBboxes)
                        
                        #image_h, image_w, _ = rotatedImage.shape
                        coor = bestScoreBboxes[0][0,0,:].astype(np.int32) # y1,x1,y2,x2
                        #print("bbox {0}".format(coor))
                        # coor[0] = int(coor[0] * image_h)
                        # coor[2] = int(coor[2] * image_h)
                        # coor[1] = int(coor[1] * image_w)
                        # coor[3] = int(coor[3] * image_w)

                        #print("rotated shape {0}".format(rotatedImage.shape))
                        extracted = rotatedImage[coor[0]:coor[2], coor[1]:coor[3], :]
                        return (annotatedImage, extracted, bestScore, bestRotation)
                        #return (None, None, None, None)
                    else:
                        return (None, None, None, None)
                
                if job['pet'] == "cat":
                    targetClass = 15
                elif job['pet'] == "dog":
                    targetClass = 16
                else:
                    log("{0}: unkown pet type: {1}".format(uid, job['pet']))
                    continue

                
                async def task(npImage, targetClass):
                    await imageProcessingSemaphore.acquire()
                    try:
                        return await FindBestRotation(npImage, targetClass)
                    finally:
                        imageProcessingSemaphore.release()
                tasks = []
                for imageNp in imagesNp:
                    tasks.append(asyncio.create_task(task(imageNp, targetClass)))

                foundPetImages = []
                annotatedImages = []
                bestScores = []
                rotations = []
                for coro in asyncio.as_completed(tasks):
                    (annotatedImage, extracted, bestScore, bestRotation) = await coro
                    if annotatedImage is None:
                        log("Did not find pet of proper type on one of the photos")
                        continue
                    log("{0}: Found {1} with confidence {2:.4f} at rotation {3}".format(uid, job['pet'], bestScore, bestRotation))
                    foundPetImages.append(extracted)
                    annotatedImages.append(annotatedImage)
                    bestScores.append("{0:.4f}".format(bestScore))
                    rotations.append(bestRotation)
                
                log("{0}: Detected needed pet ({2}) on {1} images out of {3} initial images".format(uid, len(rotations), job['pet'], len(imagesNp)))

                job["annotatedImages"] = kafkaJobQueue.imagesNpToStrList(annotatedImages)
                job["detectedPetImages"] = kafkaJobQueue.imagesNpToStrList(foundPetImages)
                job["detectedPetScores"] = bestScores
                job["detectedPetRotations"] = rotations

                await resultQueue.Enqueue(uid, job)
                log("{0}: Posted result in output queue".format(uid))
                worker.Commit()
                log("{0}: Commited".format(uid))
    asyncio.run(work())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
