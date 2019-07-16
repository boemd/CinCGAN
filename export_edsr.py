import tensorflow as tf
from hr_model import ResGAN
from datetime import datetime
import os
import logging
import cv2
import numpy as np


def export():

    edsr_model_to_load = 'checkpoints/edsr/20190625-1405\model.ckpt-9150'

    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/hr/{}".format(current_time)
    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass
    hr_gan_model_to_save = checkpoints_dir + '/model.ckpt'

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

    graph = tf.Graph()
    with graph.as_default():
        hr_gan = ResGAN(batch_size=1)
        EDSR_loss, G3_loss, D2_loss, fake_z = hr_gan.model()
        optimizers = hr_gan.optimize(EDSR_loss, G3_loss, D2_loss)
        summary_op = tf.summary.merge_all()
        logger.info('ResGAN initialized')

        saver = tf.train.Saver(hr_gan.EDSR.variables)

    _x = cv2.imread('../data/DIV2K/X_train/0001x4.png')
    _x = _x[0:31, 0:31]
    _x = cv2.cvtColor(_x, cv2.COLOR_BGR2RGB)/127.5 - 1
    x_val = np.zeros([1, _x.shape[0], _x.shape[1], _x.shape[2]])
    x_val[0] = _x
    _y = cv2.imread('../data/DIV2K/X_train_gt/0001x4.png')
    _y = _y[0:31, 0:31]
    _y = cv2.cvtColor(_y, cv2.COLOR_BGR2RGB)/127.5 - 1
    fake_y_val = np.zeros([1, _y.shape[0], _y.shape[1], _y.shape[2]])
    fake_y_val[0] = _y
    _z = cv2.imread('../data/DIV2K/Z_train/0401.png')
    _z = _z[0:127, 0:127]
    _z = cv2.cvtColor(_z, cv2.COLOR_BGR2RGB)/127.5 - 1
    fake_z_val = np.zeros([1, _z.shape[0], _z.shape[1], _z.shape[2]])
    fake_z_val[0] = _z

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, edsr_model_to_load)
        _, EDSR_loss_val, G3_loss_val, D2_loss_val, summary = (
            sess.run(
                [optimizers, EDSR_loss, G3_loss, D2_loss, summary_op],
                feed_dict={hr_gan.x: x_val,
                           hr_gan.y: fake_y_val,
                           hr_gan.fake_z: fake_z_val}
            )
        )
        save_path = saver.save(sess, hr_gan_model_to_save, global_step=0)
        logger.info("Model saved in file: %s" % save_path)


def main(unused_argv):
    export()


if __name__ == '__main__':
    tf.app.run()
