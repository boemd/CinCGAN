import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from lr_model import CleanGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints/lr/20190625-1219', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', 'g1.pb', 'XtoY model name, default: g1.pb')
tf.flags.DEFINE_string('YtoX_model', 'g2.pb', 'YtoX model name, default: g2.pb')
tf.flags.DEFINE_integer('image_size', None, 'image size, default: None')


def export_graph(model_name, XtoY=True):
    graph = tf.Graph()

    with graph.as_default():
        clean_gan = CleanGAN()
        input_image = tf.placeholder(tf.uint8, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
        input_image = utils.convert2float(input_image)
        clean_gan.model()
        if XtoY:
            output_image = clean_gan.G1.sample(tf.expand_dims(input_image, 0))
            output_image_float = clean_gan.G1.sample_f(tf.expand_dims(input_image, 0))
        else:
            output_image = clean_gan.G2.sample(tf.expand_dims(input_image, 0))
            output_image_float = clean_gan.G2.sample_f(tf.expand_dims(input_image, 0))

        output_image = tf.identity(output_image, name='output_image')
        output_image_float = tf.identity(output_image_float, name='output_image_float')
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        #latest_ckpt = tf.train.load_checkpoint(FLAGS.checkpoint_dir+'/model.ckpt-150000')
        restore_saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name, output_image_float.op.name])
        tf.train.write_graph(output_graph_def, FLAGS.checkpoint_dir, model_name, as_text=False)


def main(unused_argv):
    print('Export XtoY model...')
    export_graph(FLAGS.XtoY_model, XtoY=True)
    print('Export YtoX model...')
    export_graph(FLAGS.YtoX_model, XtoY=False)


if __name__ == '__main__':
    tf.app.run()
