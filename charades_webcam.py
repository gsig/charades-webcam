# code borrowed from datitran/object_detector_app

import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the classification.
PATH_TO_CKPT = os.path.join(CWD_PATH, 'frozen_model.pb')
PREDICTION_DECAY = 0.6 # [0,1) How slowly to update the predictions (0.99 is slowest, 0 is instant)


def loadlabels():
    # List of the strings that is used to add correct label for each box.
    labels = {}
    with open('labels.txt') as f:
        for line in f:
            x = line.split(' ')
            cls, rest = x[0], ' '.join(x[1:]).strip()
            clsint = int(cls[1:])
            labels[clsint] = {'id': clsint, 'name': rest}
    return labels
category_index = loadlabels()


def prepare_im(image_np):
    # Normalize image and fix dimensions
    image_np = cv2.resize(image_np, dsize=(224,224)).astype(np.float32)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np-mean)/std
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return image_np_expanded
    
    
def recognize_activity(image_np, sess, detection_graph, accumulator):
    image_np_expanded = prepare_im(image_np)
    image_tensor = detection_graph.get_tensor_by_name('input_image:0')
    classes = detection_graph.get_tensor_by_name('classifier/Reshape:0')

    # Actual detection.
    (classes) = sess.run(
        [classes],
        feed_dict={image_tensor: image_np_expanded})
    
    classes = np.exp(np.squeeze(classes))
    classes = classes / np.sum(classes)
    accumulator[:] = PREDICTION_DECAY*accumulator[:] + (1-PREDICTION_DECAY)*classes
    scores = np.sort(accumulator)[::-1][:3]
    classes = np.argsort(accumulator)[::-1][:3]
    boxes = np.array([[0.1,0,0.1,0], [0.2,0,0.2,0], [0.3,0,0.3,0]])
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes.astype(np.int32),
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0,
        display_score=False)
    return image_np
	

def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    accumulator = np.zeros(157,)
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(recognize_activity(frame_rgb, sess, detection_graph, accumulator))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

    c = 0
    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_rgb)
        cv2.imwrite('output/{:06d}.png'.format(c),output_rgb)
        c += 1
        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
