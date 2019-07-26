import tensorflow as tf
from cycle_in_cycle.model import CinCGAN
from datetime import datetime
import os
import helpers.utils as utils
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
tf.flags.DEFINE_string('test_set', '../../data/DIV2K/X_validation/', 'validation set')
tf.flags.DEFINE_string('ground_truth', '../../data/DIV2K/Z_test/', 'validation set')
tf.flags.DEFINE_string('model', '../checkpoints/joint/20190723-1638/model.ckpt-0', 'folder of the saved complete model')
tf.flags.DEFINE_string('output_folder', '../../data/inference/', 'output images folder')


def inference():
    if FLAGS.model is not None:
        checkpoints_dir = FLAGS.model
        checkpoints_dir = checkpoints_dir.split('/')
        checkpoints_dir.pop()
        checkpoints_dir = '/'.join(checkpoints_dir) + '/'
    else:
        return

    graph = tf.Graph()
    with graph.as_default():
        cin = CinCGAN()

        G1_loss, EDSR_loss, G3_loss, D2_loss, val_y, val_z, fake_y, fake_z = cin.model()
        optimizers = cin.optimize(G1_loss, EDSR_loss, G3_loss, D2_loss)

        lr_variables = cin.G1.variables + cin.G2.variables + cin.D1.variables
        hr_variables = cin.EDSR.variables + cin.G3.variables + cin.D2.variables
        tot_variables = lr_variables + hr_variables

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.model)
        step = int(FLAGS.model.split('-')[2])
        logging.info('Starting from a pre-trained model. Step: {}.'.format(step))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        logging.info('CinCGAN initialized.')

        try:
            ps = 0

            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            output_folder = FLAGS.output_folder + "/{}".format(current_time)
            try:
                makedirs(output_folder)
            except error:
                pass

            files = [f for f in listdir(FLAGS.test_set) if isfile(join(FLAGS.test_set, f))]
            gt_files = [f for f in listdir(FLAGS.ground_truth) if isfile(join(FLAGS.ground_truth, f))]
            rounds = len(files)
            logging.info('Inference...')

            for i in range(rounds):
                filename = FLAGS.test_set + files[i]
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im1 = np.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
                im1[0] = img
                im1 = im1.astype('uint8')

                gt = cv2.imread(FLAGS.ground_truth + gt_files[i])
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

                z = val_z.eval(feed_dict={cin.val_x: im1})
                z = z[0]

                to_save = cv2.cvtColor(z, cv2.COLOR_RGB2BGR)
                name = output_folder + '/' + files[i]
                cv2.imwrite(name, to_save)

                psnr_i = psnr(z, gt)

                print('Elaborated file {0:3d}/{1:3d}.'.format(i + 1, rounds), '  PSNR:{0:2.4f}'.format(psnr_i))

                ps += psnr_i
            ps /= rounds
            logging.info('Validation completed.')
            logging.info('Average PSNR: {:f}'.format(ps))
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def psnr(image_a, image_b):
    e = image_a.astype("double") / 255 - image_b.astype("double") / 255
    n = image_a.shape[0] * image_a.shape[1] * image_a.shape[2]
    return round(10 * math.log10(n / np.sum(np.power(e, 2))), 4)


def main(unused_argv):
    inference()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
