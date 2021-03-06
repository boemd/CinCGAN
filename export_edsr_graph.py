import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from nets.edsr import EDSR
from models.edsr_model import E_mod
import helpers.utils as utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints/edsr/20190731-1005', 'checkpoints directory path')
tf.flags.DEFINE_string('model', 'edsr.pb', 'Model name, default: edsr.pb')


def export_graph(model_name):
    graph = tf.Graph()

    with graph.as_default():
        #edsr = EDSR(name='super_res', is_training=True)
        e_mod = E_mod()
        e_mod.model()
        input_image = tf.placeholder(tf.uint8, shape=[None, None, 3], name='input_image')
        input_image = utils.convert2float(input_image)
        output_image = e_mod.edsr.sample(input_img=tf.expand_dims(input_image, 0))
        output_image = tf.identity(output_image, name='output_image')
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        restore_saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [output_image.op.name])
        tf.train.write_graph(output_graph_def, FLAGS.checkpoint_dir, model_name, as_text=False)


def main(unused_argv):
    print('Exporting model...')
    export_graph(FLAGS.model)


if __name__ == '__main__':
    tf.app.run()
