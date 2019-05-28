import tensorflow as tf
from edsr import EDSR
from datetime import datetime
import os
import logging


'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="2";
'''
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_integer('max_iter', 40000, 'maximum number of iterations during training, default: 40000')
tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 16')
tf.flags.DEFINE_string('train_file', '../data/tfrecords/train_z.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/train_z.tfrecords')

def train():
    if FLAGS.load_model is not None:
        # load the specified model
        checkpoints_dir = "checkpoints/edsr/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        # create checkpoint directory
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/edsr/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    graph = tf.Graph()
    with graph.as_default():
        edsr = EDSR('super_res', is_training=True)
        loss = edsr.model(FLAGS.train_file, FLAGS.batch_size)
        optimizers = edsr.optimize(loss)
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
                    _, loss_val, summary = sess.run([optimizers, loss, summary_op])

                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                    if step % 1000 == 0:
                        logging.info('-----------Step %d:-------------' % step)
                        logging.info('  loss   : {}'.format(loss_val))

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

