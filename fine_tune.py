import tensorflow as tf

from lr_model import CleanGAN
from hr_model import ResGAN
from datetime import datetime
import os
import utils
import logging
from os import listdir, makedirs, error
from os.path import isfile, join
import cv2
import numpy as np
import math

FLAGS = tf.flags.FLAGS

# losses flags
tf.flags.DEFINE_float('b1', 10, 'weight for cycle consistency loss, default: 10')
tf.flags.DEFINE_float('b2', 5, 'weight for identity loss, default: 5')
tf.flags.DEFINE_float('b3', 0.5, 'weight for total variation loss, default: 0.5')
tf.flags.DEFINE_float('l1', 10, 'weight for cycle consistency loss, default: 10')
tf.flags.DEFINE_float('l2', 5, 'weight for identity loss, default: 5')
tf.flags.DEFINE_float('l3', 2, 'weight for total variation loss, default: 0.5')

# training flags
tf.flags.DEFINE_integer('batch_size', 2, 'batch size, default: 16')
tf.flags.DEFINE_float('learning_rate', 0.0002, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999, 'momentum term of Adam, default: 0.999')
tf.flags.DEFINE_float('epsilon', 1e-8, 'constant for numerical stability of Adam, default: 1e-8')
tf.flags.DEFINE_integer('max_iter', 400000, 'maximum number of iterations during training, default: 400000')

# dataset flags
tf.flags.DEFINE_string('X', '../data/tfrecords/train_x.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/train_x.tfrecords')
tf.flags.DEFINE_string('Y', '../data/tfrecords/train_y.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/train_y.tfrecords')
tf.flags.DEFINE_string('Z', '../data/tfrecords/train_z.tfrecords',
                       'Z tfrecords file for training, default: data/tfrecords/train_z.tfrecords')

# validation flags
tf.flags.DEFINE_bool('validate', True, 'validation flag, default: True')
tf.flags.DEFINE_string('validation_set', '../data/DIV2K/X_validation/', 'validation set')
tf.flags.DEFINE_string('validation_ground_truth', '../data/DIV2K/X_validation_gt/', 'validation ground truth set')

# load and save flags
tf.flags.DEFINE_string('load_CinCGAN_model', None, 'folder of the saved complete model')
tf.flags.DEFINE_string('load_CleanGAN_model', 'checkpoints/lr/20190625-1219/model.ckpt-8000', 'folder of the saved CinCGAN model')
tf.flags.DEFINE_string('load_ResGAN_model', 'checkpoints/hr/20190704-1021/model.ckpt-0', 'folder of the saved ResGAN model')
'''
tf.flags.DEFINE_string('load_CinCGAN_model', None, 'folder of the saved complete model')
tf.flags.DEFINE_string('load_CleanGAN_model', 'checkpoints/lr/20190625-1219/model.ckpt-8000', 'folder of the saved CinCGAN model')
tf.flags.DEFINE_string('load_ResGAN_model', 'checkpoints/hr/model.ckpt-0', 'folder of the saved ResGAN model')
'''



def fine_tune():
    combine = False
    if FLAGS.load_CinCGAN_model is not None:
        checkpoints_dir = "checkpoints/joint/" + FLAGS.load_CinCGAN_model.lstrip("checkpoints/joint")
    else:

        # create checkpoint directory
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/joint/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

        if FLAGS.load_CleanGAN_model is not None and FLAGS.load_ResGAN_model is not None:
            checkpoints_LR_dir = "checkpoints/lr/" + FLAGS.load_CleanGAN_model.lstrip("checkpoints/lr/")
            checkpoints_HR_dir = "checkpoints/hr/" + FLAGS.load_ResGAN_model.lstrip("checkpoints/hr/")
            combine = True

    #create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(checkpoints_dir + '/LOG.log')
    handler.setLevel(logging.INFO)

    # create a logger format
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s -- %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    g1 = tf.Graph()
    with g1.as_default():
        lr_gan = CleanGAN(
            X_train_file=FLAGS.X,
            Y_train_file=FLAGS.Y,
            batch_size=FLAGS.batch_size,
            b1=FLAGS.b1,
            b2=FLAGS.b2,
            b3=FLAGS.b3,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            epsilon=FLAGS.epsilon
        )

        G1_loss, G2_loss, D1_loss, val_y, x, fake_y = lr_gan.model()
        lr_gan_optimizers = lr_gan.optimize(G1_loss, G2_loss, D1_loss)
        lr_variables = lr_gan.G1.variables + lr_gan.G2.variables + lr_gan.D1.variables

        hr_gan = ResGAN(
            Y_train_file=FLAGS.Y,
            Z_train_file=FLAGS.Z,
            batch_size=FLAGS.batch_size,
            l1=FLAGS.l1,
            l2=FLAGS.l2,
            l3=FLAGS.l3,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            epsilon=FLAGS.epsilon
        )

        EDSR_loss, G3_loss, D2_loss, fake_z = hr_gan.model()
        hr_gan_optimizers = hr_gan.optimize(EDSR_loss, G3_loss, D2_loss)
        hr_variables = hr_gan.EDSR.variables + hr_gan.G3.variables + hr_gan.D2.variables

        tot_variables =lr_variables + hr_variables
        copies = check_copies(tot_variables)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, g1)
        saver = tf.train.Saver(tot_variables)

    with tf.Session(graph=g1) as sess:
        if FLAGS.load_CinCGAN_model is not None:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.load_CinCGAN_model)
            # get step
            step = int(FLAGS.load_CinCGAN_model.split('-')[2])
            logger.info('Starting from a pre-trained model. Step: {}.'.format(step))
        elif combine:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.load_CleanGAN_model)
            saver.restore(sess, FLAGS.load_ResGAN_model)
            step = 0
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        logger.info('CinCGAN initialized.')

        try:
            print_total_parameters(logger)
            ps = 0

            dum_x = dum(FLAGS.batch_size, 32, 32, 3)
            dum_y = dum(FLAGS.batch_size, 32, 32, 3)
            dum_z = dum(FLAGS.batch_size, 32*4, 32*4, 3)

            while (not coord.should_stop()) and step <= FLAGS.max_iter:
                fake_y_val_0 = fake_y.eval()
                # fake_y_val = sess.run(fake_y)
                _, G1_loss_val, G2_loss_val, D1_loss_val, x_val, fake_y_val, summary = (
                    sess.run(
                        [lr_gan_optimizers, G1_loss, G2_loss, D1_loss, x, fake_y, summary_op],
                        feed_dict={lr_gan.fake_y: fake_y_val_0,
                                   lr_gan.psnr_validation: ps,
                                   hr_gan.x: dum_x,         # disgusting
                                   hr_gan.y: dum_y,
                                   hr_gan.fake_z: dum_z
                                   }
                    )
                ) #questa cosa va a ottimizzare anche i parametri di edsr, g3, d2?

                x_val = ((x_val + 1) / 2)*255
                fake_y_val = ((fake_y_val + 1) / 2)*255

                fake_z_val = fake_z.eval(feed_dict={hr_gan.y: fake_y_val})
                _, EDSR_loss_val, G3_loss_val, D2_loss_val, summary = (
                    sess.run(
                        [hr_gan_optimizers, EDSR_loss, G3_loss, D2_loss, summary_op],
                        feed_dict={lr_gan.fake_y: fake_y_val_0,
                                   lr_gan.psnr_validation: ps,
                                   hr_gan.x: x_val,
                                   hr_gan.y: fake_y_val,
                                   hr_gan.fake_z: fake_z_val}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 1000 == 0:
                    logger.info('-----------Step %d:-------------' % step)
                    logger.info('  G1_loss   : {}'.format(G1_loss_val))
                    logger.info('  G2_loss   : {}'.format(G2_loss_val))
                    logger.info('  EDSR_loss   : {}'.format(EDSR_loss_val))
                    logger.info('  G3_loss   : {}'.format(G3_loss_val))
                    logger.info('  D1_loss   : {}'.format(D1_loss_val))
                    logger.info('  D2_loss   : {}'.format(D2_loss_val))

                if step % 10000 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logger.info("Model saved in file: %s" % save_path)

                step += 1
        except KeyboardInterrupt:
            logger.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logger.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

        train_writer.add_summary(summary, step)
        train_writer.flush()


def write_config_file(checkpoints_dir):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(checkpoints_dir + '/config.txt', 'w') as c:
        c.write('CinCGAN' + '\n')
        c.write(date_time + '\n')
        c.write('Batch size:' + str(FLAGS.batch_size) + '\n')
        c.write('Iterations:' + str(FLAGS.max_iter) + '\n')


def check_copies(vars):
    names = []
    for var in vars:
        names.append(var.name)

    names2 = list(dict.fromkeys(names))
    if len(names) == len(names2):
        return False
    else:
        return True


def print_total_parameters(logger):
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    logger.info('# Total parameters of the network: ', total_parameters, '#')


def dum(b, h, w, c):
    return np.empty([b, h, w, c])


def main(unused_argv):
    fine_tune()


if __name__ == '__main__':
    tf.app.run()
