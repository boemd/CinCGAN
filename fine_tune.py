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


def fine_tune():
    if FLAGS.load_model is not None:
        # load the specified model
        checkpoints_dir = "checkpoints/joint/" + FLAGS.load_model.lstrip("checkpoints/joint")
    else:
        # create checkpoint directory
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/joint/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    g1 = tf.Graph()
    g2 = tf.Graph()

    with g1.as_default():
        lr_gan = CleanGAN()
        G1_loss, G2_loss, D1_loss, val_y, x, fake_y = lr_gan.model()
        lr_gan_optimizers = lr_gan.optimize(G1_loss, G2_loss, D1_loss)
        summary_op = tf.summary.merge_all()

    with tf.Session(graph=g1) as sess:
        fake_y_val = fake_y.eval()
        _, G1_loss_val, G2_loss_val, D1_loss_val, summary = (
            sess.run(
                [lr_gan_optimizers, G1_loss, G2_loss, D1_loss, summary_op],
                feed_dict={lr_gan.fake_y: fake_y_val}
            )
        )
        sess.close()

    with g2.as_default():
        hr_gan = ResGAN()
        EDSR_loss, G3_loss, D2_loss, fake_z = hr_gan.model()
        hr_gan_optimizers = hr_gan.optimize(EDSR_loss, G3_loss, D2_loss)

    with tf.Session(graph=g2) as sess:
        # restore
        # run and optimize the second gan
        sess.close()




def main(unused_argv):
    fine_tune()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
