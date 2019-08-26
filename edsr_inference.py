import tensorflow as tf
from datetime import datetime
import math
import numpy as np
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'checkpoints/edsr/20190727-1728/edsr.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input_folder', '../data/DIV2K/Z_test/', 'input image path (.png)')
tf.flags.DEFINE_string('output_folder', '../data/inference/', 'output images folder')
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
    avg_psnr = 0
    avg_ssim = 0

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
            img = cv2.resize(img, dsize=None, fx=1 / FLAGS.scale, fy=1 / FLAGS.scale, interpolation=cv2.INTER_LINEAR)
            im1 = np.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
            im1[0] = img
            im1 = im1.astype('uint8')
            generated = output_image.eval(feed_dict={input_image: img})
            psnr_ev = psnr(gt, generated)
            ssim_ev = ssim(gt, generated)
            avg_psnr += psnr_ev
            avg_ssim += ssim_ev
            print('Elaborated file {0:2d}/{1:3d}.'.format(i + 1, len(files)),
                  '  PSNR:{0:2.4f}'.format(psnr_ev),
                  '  SSIM:{0:2.4f}'.format(ssim_ev))
            to_write = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)
            out_name = output_folder + '/' + files[i][0:-4] + '.png'
            cv2.imwrite(out_name, to_write)

        avg_psnr /= len(files)
        avg_ssim /= len(files)
        print('Average PSNR: ', avg_psnr)
        print('Average SSIM: ', avg_ssim)


def psnr(imageA, imageB):
    E = imageA.astype("double")/255 - imageB.astype("double")/255
    N = imageA.shape[0] * imageA.shape[1] * imageA.shape[2]
    return 10 * math.log10(N / np.sum(np.power(E, 2)))


def ssim(im1, im2):
    h, w, d = im1.shape
    ssim = 0
    for i in range(d):
        a = im1[:, :, i]
        b = im2[:, :, i]
        K = [0.01, 0.03]
        L = 255

        C1 = (K[0]*L)**2
        C2 = (K[1]*L)**2
        a = a.astype(float)
        b = b.astype(float)
        mu1 = cv2.GaussianBlur(a, (11, 11), 1.5, cv2.BORDER_ISOLATED)
        mu1 = mu1[5:, 5:]
        mu1 = mu1[:-5, :-5]
        mu2 = cv2.GaussianBlur(b.astype(float), (11, 11), 1.5, cv2.BORDER_ISOLATED)
        mu2 = mu2[5:, 5:]
        mu2 = mu2[:-5, :-5]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = np.multiply(mu1, mu2)

        sigma1_sq = cv2.GaussianBlur(a**2, (11, 11), 1.5)
        sigma1_sq = sigma1_sq[5:, 5:]
        sigma1_sq = sigma1_sq[:-5, :-5] - mu1_sq

        sigma2_sq = cv2.GaussianBlur(b**2, (11, 11), 1.5)
        sigma2_sq = sigma2_sq[5:, 5:]
        sigma2_sq = sigma2_sq[:-5, :-5] - mu2_sq

        sigma12 = cv2.GaussianBlur(np.multiply(a, b), (11, 11), 1.5)
        sigma12 = sigma12[5:, 5:]
        sigma12 = sigma12[:-5, :-5] - mu1_mu2

        if C1 > 0 and C2 > 0:
            ssim_map = np.divide(np.multiply((2 * mu1_mu2 + C1), (2 * sigma12 + C2)), np.multiply((mu1_sq + mu2_sq + C1), (sigma1_sq + sigma2_sq + C2)))
        else:
            # this is useless
            numerator1 = 2 * mu1_mu2 + C1
            numerator2 = 2 * sigma12 + C2
            denominator1 = mu1_sq + mu2_sq + C1
            denominator2 = sigma1_sq + sigma2_sq + C2
            ssim_map = np.ones((h, w))
            index = np.nonzero(np.clip(np.dot(denominator1, denominator2), a_min=0))
            ssim_map[index] = np.dot(numerator1[index], numerator2[index]) / \
                              np.dot(denominator1[index], denominator2[index])
            index = np.nonzero(denominator1) and np.argwhere(denominator2 == 0)
            ssim_map[index] = numerator1[index] / denominator1[index]
        ssim += np.mean(ssim_map)
    ssim /= d
    return np.round(ssim, 4)


if __name__ == '__main__':
    tf.app.run()
