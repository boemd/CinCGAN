import tensorflow as tf
from hr_model import ResGAN
from datetime import datetime
import os
import logging


def export():
    edsr_model_to_load = 'checkpoints/edsr/20190625-1405\model.ckpt-9150'
    checkpoints_dir = 'checkpoints/edsr'
    hr_gan_model_to_save = 'checkpoints/hr/model.ckpt'

    graph = tf.Graph()
    with graph.as_default():
        hr_gan = ResGAN()
        EDSR_loss, G3_loss, D2_loss, fake_z = hr_gan.model()
        optimizers = hr_gan.optimize(EDSR_loss, G3_loss, D2_loss)
        logging.info('ResGAN initialized')

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver(hr_gan.EDSR.variables)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        #latest_ckpt = tf.train.latest_checkpoint(checkpoints_dir)
        saver.restore(sess, edsr_model_to_load)

        save_path = saver.save(sess, hr_gan_model_to_save, global_step=0)
        logging.info("Model saved in file: %s" % save_path)


def main(unused_argv):
    export()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()