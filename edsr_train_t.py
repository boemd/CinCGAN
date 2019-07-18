import tensorflow as tf
from model_edsr import E_mod
from datetime import datetime
import os
import utils
import logging
from os import listdir, makedirs, error
from os.path import isfile, join
import cv2
import numpy as np
import math

'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="2";
'''
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 10')
tf.flags.DEFINE_integer('scale', 4, 'scale, default: 4')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999, 'momentum term of Adam, default: 0.999')
tf.flags.DEFINE_float('epsilon', 1e-8, 'constant for numerical stability of Adam, default: 1e-8')
tf.flags.DEFINE_string('Z', '../data/tfrecords/train_z.tfrecords',
                       'Z tfrecords file for training, default: data/tfrecords/train_z.tfrecords')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. checkpoints/edsr/20190625-1405), default: None')
tf.flags.DEFINE_integer('max_iter', 1000000, 'maximum number of iterations during training, default: 400000')
tf.flags.DEFINE_string('validation_set', '../data/DIV2K/Z_test/', 'validation set')
tf.flags.DEFINE_boolean('validate', True, 'validation flag, default: True')


def train():
    if FLAGS.load_model is not None:
        # load the specified model
        checkpoints_dir = "checkpoints/edsr/" + FLAGS.load_model.lstrip("checkpoints/edsr")
    else:
        # create checkpoint directory
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/edsr/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass
    print('#'*50)
    print(checkpoints_dir)

    graph = tf.Graph()
    with graph.as_default():
        ed = E_mod(train_file=FLAGS.Z, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate, scale=FLAGS.scale)
        loss, val_y = ed.model()
        optimizer = ed.optimize(loss, FLAGS.learning_rate)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        flag_resume = False
        if FLAGS.load_model is not None:
            sess.run(tf.global_variables_initializer())
            latest_ckpt = tf.train.latest_checkpoint(checkpoints_dir)
            saver.restore(sess, latest_ckpt)
            
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            step = int(meta_graph_path.split("-")[2].split(".")[0])
            flag_resume = True
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        logging.info('EDSR_trainer initialized.')
        logging.info('Starting from step {}'.format(step))

        try:
            print_total_parameters()
            ps = 0
            while (not coord.should_stop()) and step <= FLAGS.max_iter:
                if flag_resume or step == FLAGS.max_iter:
                    flag_resume = False
                    validate(ed, val_y, sess)

                _, loss_val, summary = sess.run([optimizer, loss, summary_op], feed_dict={ed.psnr_validation: ps})

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 1000 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  loss   : {}'.format(loss_val))

                if step % 10000 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)
                    if FLAGS.validate:
                        ps = validate(ed, val_y, sess)

                step += 1
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def validate(ed, val_y, sess):
    files = [f for f in listdir(FLAGS.validation_set) if isfile(join(FLAGS.validation_set, f))]
    rounds = len(files)
    logging.info('Validating...')
    ps = 0
    for i in range(rounds):
        img = cv2.imread(FLAGS.validation_set + files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = img
        img = cv2.resize(img, dsize=None, fx=1 / FLAGS.scale, fy=1 / FLAGS.scale, interpolation=cv2.INTER_CUBIC)
        im1 = np.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
        im1[0] = img
        im1 = im1.astype('uint8')
        y = val_y.eval(feed_dict={ed.val_x: im1})
        y = y[0]
        ps += psnr(y, gt)
    ps /= rounds
    logging.info('Validation completed. PSNR: {:f}'.format(ps))
    return ps


def psnr(imageA, imageB):
    E = imageA.astype("double")/255 - imageB.astype("double")/255
    N = imageA.shape[0] * imageA.shape[1] * imageA.shape[2]
    return round(10 * math.log10(N / np.sum(np.power(E, 2))), 4)


def print_total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('# Total parameters of the network: ', total_parameters, '#')


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
