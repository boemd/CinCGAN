import tensorflow as tf
from model import CinCGAN
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

# loss parameters
tf.flags.DEFINE_float('b1', 10, 'weight for the lr cycle consistency loss, default: 10')
tf.flags.DEFINE_float('b2', 5, 'weight for the lr identity loss, default: 5')
tf.flags.DEFINE_float('b3', 0.5, 'weight for the lr total variation loss, default: 0.5')
tf.flags.DEFINE_float('l1', 10, 'weight for the hr cycle consistency loss, default: 10')
tf.flags.DEFINE_float('l2', 5, 'weight for the hr identity loss, default: 5')
tf.flags.DEFINE_float('l3', 2, 'weight for the hr total variation loss, default: 0.5')

# learning parameters
tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 16')
tf.flags.DEFINE_float('learning_rate', 0.0002, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999, 'momentum term of Adam, default: 0.999')
tf.flags.DEFINE_float('epsilon', 1e-8, 'constant for numerical stability of Adam, default: 1e-8')
tf.flags.DEFINE_integer('max_iter', 400000, 'maximum number of iterations during training, default: 400000')
tf.flags.DEFINE_integer('scale', 4, 'scale of the super-resolution model, default: 4')

# dataset parameters
#  training
tf.flags.DEFINE_string('X', '../data/tfrecords/train_x.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/train_x.tfrecords')
tf.flags.DEFINE_string('Y', '../data/tfrecords/train_y.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/train_y.tfrecords')
tf.flags.DEFINE_string('Z', '../data/tfrecords/train_z.tfrecords',
                       'Z tfrecords file for training, default: data/tfrecords/train_z.tfrecords')
#  validation
tf.flags.DEFINE_string('validation_set', '../data/DIV2K/X_validation/', 'validation set')
tf.flags.DEFINE_string('validation_ground_truth_y', '../data/DIV2K/Y_validation/', 'validation ground truth set')
tf.flags.DEFINE_string('validation_ground_truth_z', '../data/DIV2K/Z_validation/', 'validation ground truth set')


# pre-trained models
tf.flags.DEFINE_string('load_CinCGAN_model', 'checkpoints/joint/20190718-1146/model.ckpt-100', 'folder of the saved complete model')
tf.flags.DEFINE_string('load_CleanGAN_model', None, 'folder of the saved CinCGAN model')
tf.flags.DEFINE_string('load_EDSR_model', None, 'folder of the saved EDSR model')
'''
tf.flags.DEFINE_string('load_CinCGAN_model', 'checkpoints/joint/20190718-1146/model.ckpt-100', 'folder of the saved complete model')
tf.flags.DEFINE_string('load_CleanGAN_model', 'checkpoints/lr/20190624-1117-main/model.ckpt-50000', 'folder of the saved CinCGAN model')
tf.flags.DEFINE_string('load_EDSR_model', 'checkpoints/edsr/20190718-1028/model.ckpt-2000', 'folder of the saved EDSR model')
'''

# others
tf.flags.DEFINE_bool('validate', False, 'validation flag, default: True')
tf.flags.DEFINE_bool('save_samples', False, 'samples flag, default: False')


def train():
    combine = False
    if FLAGS.load_CinCGAN_model is not None:
        #checkpoints_dir = "checkpoints/joint/" + FLAGS.load_CinCGAN_model.lstrip("checkpoints/joint")
        checkpoints_dir = FLAGS.load_CinCGAN_model
        checkpoints_dir = checkpoints_dir.split('/')
        checkpoints_dir.pop()
        checkpoints_dir = '/'.join(checkpoints_dir) + '/'
    else:

        # create checkpoint directory
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/joint/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

        if FLAGS.load_CleanGAN_model is not None and FLAGS.load_EDSR_model is not None:
            #checkpoints_LR_dir = "checkpoints/lr/" + FLAGS.load_CleanGAN_model.lstrip("checkpoints/lr/")
            checkpoints_LR_dir = FLAGS.load_CleanGAN_model
            checkpoints_LR_dir = checkpoints_LR_dir.split('/')
            checkpoints_LR_dir.pop()
            checkpoints_LR_dir = '/'.join(checkpoints_LR_dir) + '/'

            #checkpoints_HR_dir = "checkpoints/hr/" + FLAGS.load_EDSR_model.lstrip("checkpoints/hr/")
            checkpoints_HR_dir = FLAGS.load_EDSR_model
            checkpoints_HR_dir = checkpoints_HR_dir.split('/')
            checkpoints_HR_dir.pop()
            checkpoints_HR_dir = '/'.join(checkpoints_HR_dir) + '/'


            combine = True

    # create a logger
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
    write_config_file(checkpoints_dir)

    graph = tf.Graph()
    with graph.as_default():
        cin = CinCGAN(
            X_train_file=FLAGS.X,
            Y_train_file=FLAGS.Y,
            Z_train_file=FLAGS.Z,
            batch_size=FLAGS.batch_size,
            scale=FLAGS.scale,
            b1=FLAGS.b1,
            b2=FLAGS.b2,
            b3=FLAGS.b3,
            l1=FLAGS.l1,
            l2=FLAGS.l2,
            l3=FLAGS.l3,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            epsilon=FLAGS.epsilon
        )

        G1_loss, G2_loss, D1_loss, EDSR_loss, G3_loss, D2_loss, val_y, val_z, fake_y, fake_z = cin.model()
        optimizers = cin.optimize(G1_loss, G2_loss, D1_loss, EDSR_loss, G3_loss, D2_loss)

        lr_variables = cin.G1.variables + cin.G2.variables + cin.D1.variables
        hr_variables = cin.EDSR.variables + cin.G3.variables + cin.D2.variables
        tot_variables = lr_variables + hr_variables


        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

        saver_lr = tf.train.Saver(lr_variables)
        saver_edsr = tf.train.Saver(cin.EDSR.variables)

    flag_resume = False
    with tf.Session(graph=graph) as sess:
        if FLAGS.load_CinCGAN_model is not None:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.load_CinCGAN_model)
            # get step
            step = int(FLAGS.load_CinCGAN_model.split('-')[2])
            logger.info('Starting from a pre-trained model. Step: {}.'.format(step))
        elif combine:
            sess.run(tf.global_variables_initializer())
            saver_lr.restore(sess, FLAGS.load_CleanGAN_model)
            saver_edsr.restore(sess, FLAGS.load_EDSR_model)
            step = 0
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        logger.info('CinCGAN initialized.')

        try:
            # print_total_parameters(logger)

            ps_y = 0  # validation PSNR for the images generated by the inner CycleGAN
            ps_z = 0  # validation PSNR for the images generated by the outer CycleGAN

            # Freeze G1 and D1
            cin.G2.is_training = False
            cin.D1.is_training = False

            while (not coord.should_stop()) and step <= FLAGS.max_iter:

                if flag_resume or step == FLAGS.max_iter:
                    flag_resume = False
                    #validate(logger, cin, val_y, val_z)

                fake_y_val = fake_y.eval()
                fake_z_val = fake_z.eval()
                _, G1_loss_val, G2_loss_val, D1_loss_val, EDSR_loss_val, G3_loss_val, D2_loss_val, summary = sess.run(
                    [optimizers, G1_loss, G2_loss, D1_loss, EDSR_loss, G3_loss, D2_loss, summary_op],
                    feed_dict={cin.fake_y: fake_y_val,
                               cin.fake_z: fake_z_val,
                               cin.psnr_validation_y: ps_y,
                               cin.psnr_validation_z: ps_z
                               }
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 10 == 0:
                    print('-----------Step %d:-------------' % step)
                    print('  G1_loss     : {}'.format(G1_loss_val))
                    print('  G2_loss     : {}'.format(G2_loss_val))
                    print('  D1_loss     : {}'.format(D1_loss_val))
                    print('  EDSR_loss   : {}'.format(EDSR_loss_val))
                    print('  G3_loss     : {}'.format(G3_loss_val))
                    print('  D2_loss     : {}'.format(D2_loss_val))

                if step % 100 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)
                    if FLAGS.validate:
                        ps_y, ps_z = validate(logger, cin, val_y, val_z)

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


def validate(logger, cin, val_y, val_z):
    '''
    files = [f for f in listdir(FLAGS.validation_set) if isfile(join(FLAGS.validation_set, f))]
    gt_y_files = [f for f in listdir(FLAGS.validation_ground_truth_y) if isfile(join(FLAGS.validation_ground_truth_y, f))]
    gt_z_files = [f for f in listdir(FLAGS.validation_ground_truth_z) if isfile(join(FLAGS.validation_ground_truth_z, f))]
    rounds = len(files)
    logger.info('Validating...')
    ps_y = 0
    ps_z = 0
    for i in range(rounds):
        img = cv2.imread(FLAGS.validation_set + files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im1 = np.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
        im1[0] = img
        im1 = im1.astype('uint8')

        gt_y = cv2.imread(FLAGS.validation_ground_truth_y + gt_y_files[i])
        gt_y = cv2.cvtColor(gt_y, cv2.COLOR_BGR2RGB)
        y = val_y.eval(feed_dict={cin.val_x: im1})
        y = y[0]

        gt_z = cv2.imread(FLAGS.validation_ground_truth_z + gt_z_files[i])
        gt_z = cv2.cvtColor(gt_z, cv2.COLOR_BGR2RGB)
        z = val_z.eval(feed_dict={cin.val_x: im1})
        z = z[0]

        ps_y += psnr(y, gt_y)
        ps_z += psnr(z, gt_z)
    ps_y /= rounds
    ps_z /= rounds
    logger.info('Validation completed. PSNR: {:f}'.format(ps_y))
    return ps_y, ps_z
    '''
    return 0, 0


def psnr(imageA, imageB):
    E = imageA.astype("double")/255 - imageB.astype("double")/255
    N = imageA.shape[0] * imageA.shape[1] * imageA.shape[2]
    return round(10 * math.log10(N / np.sum(np.power(E, 2))), 4)


def write_config_file(checkpoints_dir):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(checkpoints_dir + '/config.txt', 'w') as c:
        c.write('COMPLETE MODEL' + '\n')
        c.write(date_time + '\n')
        c.write('Batch size:' + str(FLAGS.batch_size) + '\n')
        c.write('Iterations:' + str(FLAGS.max_iter) + '\n')
        c.write('Cycle consistency loss term (b1):' + str(FLAGS.b1) + '\n')
        c.write('Identity loss term (b2):' + str(FLAGS.b2) + '\n')
        c.write('Total variation loss term (b3):' + str(FLAGS.b3) + '\n')
        c.write('Cycle consistency loss term (l1):' + str(FLAGS.l1) + '\n')
        c.write('Identity loss term (l2):' + str(FLAGS.l2) + '\n')
        c.write('Total variation loss term (l3):' + str(FLAGS.l3) + '\n')


def print_total_parameters(logger):
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    logger.info('Total parameters of the network: ', total_parameters, '#')


def main(unused_argv):
    train()


if __name__ == '__main__':
    tf.app.run()

