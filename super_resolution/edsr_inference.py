import tensorflow as tf
from datetime import datetime
import math
import numpy as np
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '../checkpoints/edsr/20190723-1610/edsr.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input_folder', '../../data/DIV2K/Z_test/', 'input image path (.png)')
tf.flags.DEFINE_string('output_folder', '../../data/inference/', 'output images folder')
tf.flags.DEFINE_integer('scale', 4, 'scale, default: 4')


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def main(unused_argv):
    from os import listdir, makedirs, error
    from os.path import isfile, join

    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    output_folder = FLAGS.output_folder + "/{}".format(current_time)
    print(output_folder)
    print('#'*30)
    avg_psnr = 0

    graph = load_graph(FLAGS.model)
    input_image = graph.get_tensor_by_name('prefix/input_image:0')  # uint8
    output_image = graph.get_tensor_by_name('prefix/output_image:0')  # string

    # output_image = tf.clip_by_value(output_image, -1, 1)

    output_image = tf.image.decode_png(output_image, channels=3)  # uint8

    try:
        makedirs(output_folder)
    except error:
        pass

    files = [f for f in listdir(FLAGS.input_folder) if isfile(join(FLAGS.input_folder, f))]

    with tf.Session(graph=graph) as sess:
        for i in range(len(files)):
            img = cv2.imread(FLAGS.input_folder + files[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = img
            img = cv2.resize(img, dsize=None, fx=1 / FLAGS.scale, fy=1 / FLAGS.scale, interpolation=cv2.INTER_CUBIC)
            im1 = np.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
            im1[0] = img
            im1 = im1.astype('uint8')
            generated = output_image.eval(feed_dict={input_image: img})
            psnr_ev = psnr(gt, generated)
            avg_psnr += psnr_ev
            print('Elaborated file {0:2d}/{1:3d}.'.format(i + 1, len(files)),
                  '  PSNR:{0:2.4f}'.format(psnr_ev))
            to_write = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)
            out_name = output_folder + '/' + files[i][0:-4] + '.png'
            cv2.imwrite(out_name, to_write)

        avg_psnr /= len(files)
        print('Average PSNR: ', avg_psnr)


def psnr(imageA, imageB):
    E = imageA.astype("double")/255 - imageB.astype("double")/255
    N = imageA.shape[0] * imageA.shape[1] * imageA.shape[2]
    return 10 * math.log10(N / np.sum(np.power(E, 2)))


if __name__ == '__main__':
    tf.app.run()
