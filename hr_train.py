import tensorflow as tf
from hr_model import ResGAN
from datetime import datetime
import os
import logging
from os import listdir, makedirs, error
from os.path import isfile, join

'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="2";
'''
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 16')
tf.flags.DEFINE_float('l1', 10, 'weight for cycle consistency loss, default: 10')
tf.flags.DEFINE_float('l2', 5, 'weight for identity loss, default: 5')
tf.flags.DEFINE_float('l3', 2, 'weight for total variation loss, default: 0.5')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999, 'momentum term of Adam, default: 0.999')
tf.flags.DEFINE_float('epsilon', 1e-8, 'constant for numerical stability of Adam, default: 1e-8')
tf.flags.DEFINE_string('Y', '../data/tfrecords/train_y2.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/train_y2.tfrecords')
tf.flags.DEFINE_string('Z', '../data/tfrecords/train_z.tfrecords',
                       'Z tfrecords file for training, default: data/tfrecords/train_z.tfrecords')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_integer('max_iter', 400000, 'maximum number of iterations during training, default: 400000')


def train():
    if FLAGS.load_model is not None:
        # load the specified model
        checkpoints_dir = "checkpoints/hr/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        # create checkpoint directory
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/hr/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    write_config_file(checkpoints_dir)

    graph = tf.Graph()
    with graph.as_default():
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
        optimizers = hr_gan.optimize(EDSR_loss, G3_loss, D2_loss)
        logging.info('ResGAN initialized')

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            print_total_parameters()

            while (not coord.should_stop()) and step < FLAGS.max_iter:
                fake_z_val = fake_z.eval()
                _, EDSR_loss_val, G3_loss_val, D2_loss_val, summary = (
                    sess.run(
                        [optimizers, EDSR_loss, G3_loss, D2_loss, summary_op],
                        feed_dict={hr_gan.fake_z: fake_z_val}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 1000 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  EDSR_loss   : {}'.format(EDSR_loss_val))
                    logging.info('  G3_loss   : {}'.format(G3_loss_val))
                    logging.info('  D2_loss   : {}'.format(D2_loss_val))

                if step % 10000 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

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


def write_config_file(checkpoints_dir):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(checkpoints_dir + '/config.txt', 'w') as c:
        c.write('HIGH RESOLUTION MODEL' + '\n')
        c.write(date_time + '\n')
        c.write('Batch size:' + str(FLAGS.batch_size) + '\n')
        c.write('Iterations:' + str(FLAGS.max_iter) + '\n')
        c.write('Cycle consistency loss term (l1):' + str(FLAGS.l1) + '\n')
        c.write('Identity loss term (l2):' + str(FLAGS.l2) + '\n')
        c.write('Total variation loss term (l3):' + str(FLAGS.l3) + '\n')


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
