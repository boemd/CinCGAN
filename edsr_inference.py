import tensorflow as tf
from edsr import EDSR
import cv2
from os import listdir, makedirs, error
from os.path import isfile, join
import numpy as np

validation_set = '../data/DIV2K/X_validation/'
validation_ground_truth = '../data/DIV2K/X_validation_gt/'

files = [f for f in listdir(validation_set) if isfile(join(validation_set, f))]
gt_files = [f for f in listdir(validation_ground_truth) if isfile(join(validation_ground_truth, f))]
rounds = len(files)

graph = tf.Graph()

with graph.as_default():
    edsr = EDSR(name='super_res', is_training=True)
    input = tf.placeholder(tf.uint8, shape=[1, 339, 510, 3], name='input_image')
    input = tf.to_float(input)
    out = edsr(input)

    saver = tf.train.Saver()

checkpoints_dir = 'checkpoints/edsr/aaa'

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
    meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
    restore = tf.train.import_meta_graph(meta_graph_path)
    restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))


    for i in range(rounds):
        img = cv2.imread(validation_set + files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im1 = np.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
        im1[0] = img
        im1 = im1.astype('uint8')
        gt = cv2.imread(validation_ground_truth + gt_files[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        y = sess.run(out, feed_dict={image: im1})
        y = y[0]
        y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    l = [n.name for n in tf.get_default_graph().as_graph_def().node]
    for j in l:
        print(j+'\n')
    '''

    img = cv2.imread(validation_set + files[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im1 = np.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
    im1[0] = img
    im1 = im1.astype('uint8')
    y = sess.run(out, feed_dict={input: im1})
    y = y.astype('uint8')
    y = y[0]
    y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
