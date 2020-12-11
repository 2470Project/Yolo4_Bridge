from datetime import  datetime
import time

import dlib
import imutils
import tensorflow as tf
from imutils.video import FPS

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
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from pyimagesearch.utils import Conf



flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    #############################################3
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    conf = Conf('./config/config.json')
    # check to see if the Dropbox should be used

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(conf["prototxt_path"],
                                   conf["model_path"])
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] warming up camera...")
    # vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    H = None
    W = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=conf["max_disappear"],
                         maxDistance=conf["max_distance"])
    trackers = []
    trackableObjects = {}

    # keep the count of total number of frames
    totalFrames = 0

    # initialize the log file
    logFile = None

    # initialize the list of various points used to calculate the avg of
    # the vehicle speed
    points = [("A", "B"), ("B", "C"), ("C", "D")]

    # start the frames per second throughput estimator
    fps = FPS().start()


    ############################################
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fpss = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fpss, (width, height))

    while True:
        return_value, frame = vid.read()
        ts = datetime.now()
        f = frame
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        # fps = 1.0 / (time.time() - start_time)
        # # print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("result", result)

        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


        # resize the frame
        frame = imutils.resize(frame, width=conf["frame_width"])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]
            meterPerPixel = conf["distance"] / W

        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % conf["track_object"] == 0:
            # initialize our new set of object trackers
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(f, size=(300, 300),
                                         ddepth=cv2.CV_8U)
            net.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5,
                                                              127.5, 127.5])
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence`
                # is greater than the minimum confidence
                if confidence > conf["confidence"]:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a car, ignore it
                    if CLASSES[idx] != "car":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing
        # throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, if there is a trackable object and its speed has
            # not yet been estimated then estimate it
            elif not to.estimated:
                # check if the direction of the object has been set, if
                # not, calculate it, and set it
                if to.direction is None:
                    y = [c[0] for c in to.centroids]
                    direction = centroid[0] - np.mean(y)
                    to.direction = direction

                # if the direction is positive (indicating the object
                # is moving from left to right)
                if to.direction > 0:
                    # check to see if timestamp has been noted for
                    # point A
                    if to.timestamp["A"] == 0:
                        # if the centroid's x-coordinate is greater than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] > conf["speed_estimation_zone"]["A"]:
                            to.timestamp["A"] = ts
                            to.position["A"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point B
                    elif to.timestamp["B"] == 0:
                        # if the centroid's x-coordinate is greater than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] > conf["speed_estimation_zone"]["B"]:
                            to.timestamp["B"] = ts
                            to.position["B"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point C
                    elif to.timestamp["C"] == 0:
                        # if the centroid's x-coordinate is greater than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] > conf["speed_estimation_zone"]["C"]:
                            to.timestamp["C"] = ts
                            to.position["C"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point D
                    elif to.timestamp["D"] == 0:
                        # if the centroid's x-coordinate is greater than
                        # the corresponding point then set the timestamp
                        # as current timestamp, set the position as the
                        # centroid's x-coordinate, and set the last point
                        # flag as True
                        if centroid[0] > conf["speed_estimation_zone"]["D"]:
                            to.timestamp["D"] = ts
                            to.position["D"] = centroid[0]
                            to.lastPoint = True

                # if the direction is negative (indicating the object
                # is moving from right to left)
                elif to.direction < 0:
                    # check to see if timestamp has been noted for
                    # point D
                    if to.timestamp["D"] == 0:
                        # if the centroid's x-coordinate is lesser than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] < conf["speed_estimation_zone"]["D"]:
                            to.timestamp["D"] = ts
                            to.position["D"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point C
                    elif to.timestamp["C"] == 0:
                        # if the centroid's x-coordinate is lesser than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] < conf["speed_estimation_zone"]["C"]:
                            to.timestamp["C"] = ts
                            to.position["C"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point B
                    elif to.timestamp["B"] == 0:
                        # if the centroid's x-coordinate is lesser than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] < conf["speed_estimation_zone"]["B"]:
                            to.timestamp["B"] = ts
                            to.position["B"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point A
                    elif to.timestamp["A"] == 0:
                        # if the centroid's x-coordinate is lesser than
                        # the corresponding point then set the timestamp
                        # as current timestamp, set the position as the
                        # centroid's x-coordinate, and set the last point
                        # flag as True
                        if centroid[0] < conf["speed_estimation_zone"]["A"]:
                            to.timestamp["A"] = ts
                            to.position["A"] = centroid[0]
                            to.lastPoint = True

                # check to see if the vehicle is past the last point and
                # the vehicle's speed has not yet been estimated, if yes,
                # then calculate the vehicle speed and log it if it's
                # over the limit
                if to.lastPoint and not to.estimated:
                    # initialize the list of estimated speeds
                    estimatedSpeeds = []

                    # loop over all the pairs of points and estimate the
                    # vehicle speed
                    for (i, j) in points:
                        # calculate the distance in pixels
                        d = to.position[j] - to.position[i]
                        distanceInPixels = abs(d)

                        # check if the distance in pixels is zero, if so,
                        # skip this iteration
                        if distanceInPixels == 0:
                            continue

                        # calculate the time in hours
                        t = to.timestamp[j] - to.timestamp[i]
                        timeInSeconds = abs(t.total_seconds())
                        timeInHours = timeInSeconds / (60 * 60)

                        # calculate distance in kilometers and append the
                        # calculated speed to the list
                        distanceInMeters = distanceInPixels * meterPerPixel
                        distanceInKM = distanceInMeters / 1000
                        estimatedSpeeds.append(distanceInKM / timeInHours)

                    # calculate the average speed
                    to.calculate_speed(estimatedSpeeds)

                    # set the object as estimated
                    to.estimated = True
                    if to.direction > 0:
                        print("[INFO] The direction of the vehicle that just passed" \
                              " : left to right")
                    else:
                        print("[INFO] The direction of the vehicle that just passed" \
                              ": right to left")

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4,
                       (0, 255, 0), -1)


        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
