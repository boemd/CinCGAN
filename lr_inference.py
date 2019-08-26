import tensorflow as tf
from datetime import datetime
import math
import numpy as np
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'checkpoints/lr/20190406-1855/g1.pb', 'model path (.pb)')
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
        ps = 0
        ss = 0
        for i in range(len(files)):
            img = cv2.imread(FLAGS.input_folder + files[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('uint8')
            generated = output_image.eval(feed_dict={input_image: img})
            gt = cv2.imread(FLAGS.input_gt_folder + gt_files[i])
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            ps_i = psnr(generated, gt)
            ps_0 = psnr(img, gt)
            ss_i = ssim(img, gt)
            ps += ps_i
            ss += ss_i
            '''
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
            '''
            to_save = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)
            name = output_folder + '/' + files[i]
            cv2.imwrite(name, to_save)
            print('Elaborated file {0:3d}/{1:3d}.'.format(i + 1, len(files)),
                  '  PSNR:{0:2.4f}'.format(ps_i),
                  '  SSIM:{0:2.4f}'.format(ss_i),
                  '  PSNR Improvement: {0:1.4f}'.format(ps_i/ps_0 - 1))

        ps /= len(files)
        ss /= len(files)

        print('Average PSNR: ', ps)
        print('Average SSIM: ', ss)


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
