
import tensorflow as tf
from datetime import datetime
import math
import numpy as np

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'checkpoints/lr/20190531-1616/g1.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input_folder', '../data/DIV2K/X_validation/', 'input image path (.png)')
tf.flags.DEFINE_string('input_gt_folder', '../data/DIV2K/X_validation_gt/', 'input image path (.png)')
tf.flags.DEFINE_string('output_folder', '../data/inference/', 'output images folder')

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

    # output_image = tf.clip_by_value(t=output_image, -1, 1)

    output_image_und = output_image  # non-decoded output image
    output_image = tf.image.decode_png(output_image, channels=3)  # uint8

    try:
        makedirs(output_folder)
    except error:
        pass

    files = [f for f in listdir(FLAGS.input_folder) if isfile(join(FLAGS.input_folder, f))]
    gt_files = [f for f in listdir(FLAGS.input_gt_folder) if isfile(join(FLAGS.input_gt_folder, f))]

    with tf.Session(graph=graph) as sess:
        for i in range(len(files)):
            image_name = files[i]
            image_data = tf.gfile.FastGFile(FLAGS.input_folder + image_name, 'rb').read()
            image_data = tf.image.decode_png(image_data)
            rr = image_data.eval()
            generated = output_image.eval(feed_dict={input_image: rr})
            to_write = output_image_und.eval(feed_dict={input_image: rr})

            gt_image_name = gt_files[i]
            gt_image_data = tf.gfile.FastGFile(FLAGS.input_gt_folder + gt_image_name, 'rb').read()
            gt_image_data = tf.image.decode_png(gt_image_data)
            gt = gt_image_data.eval()

            psnr = tf.image.psnr(gt, generated, max_val=255)
            psnr_0 = tf.image.psnr(gt, rr, max_val=255)
            psnr_ev = psnr.eval()
            psnr_0_ev = psnr_0.eval()
            # plt.imshow(generated)


            #psnr_ev = psnr(gt, generated)
            #psnr_0_ev = psnr(gt, rr)
            avg_psnr += psnr_ev


            with open(output_folder + '/' + image_name[0:-4] + '.png', 'wb') as f:
                f.write(to_write)

            print('Elaborated file {0:2d}/{1:3d}.'.format(i + 1, len(files)),
                  '  PSNR:{0:2.4f}'.format(psnr_ev),
                  '  Improvement: {0:1.4f}'.format(psnr_ev/psnr_0_ev - 1))

        avg_psnr /= len(files)
        print('Average PSNR: ', avg_psnr)


def psnr(imageA, imageB):
    E = imageA.astype("double")/255 - imageB.astype("double")/255
    N = imageA.shape[0] * imageA.shape[1] * imageA.shape[2]
    return 10 * math.log10(N / np.sum(np.power(E, 2)))


if __name__ == '__main__':
    tf.app.run()
