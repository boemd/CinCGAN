import tensorflow as tf
import helpers.utils as utils
import numpy as np
import random
import cv2

H = 183
W = 279


class Reader():
    def __init__(self, tfrecords_file, image_size=None,
                 min_queue_examples=0, batch_size=2, num_threads=8, name='', crop_size=32, scale=4):
        """
        Args:
          tfrecords_file: string, tfrecords file path
          min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
          batch_size: integer, number of images per batch
          num_threads: integer, number of preprocess threads
        """
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name
        self.crop_size = crop_size
        self.scale = scale

        self.im1p = tf.placeholder(tf.uint8, shape=[None, None, 3])
        self.im2p = tf.placeholder(tf.uint8, shape=[None, None, 3])
        self.height = tf.placeholder(tf.int32)
        self.width = tf.placeholder(tf.int32)

    def b_preprocess(self, image, seed=777, crop=True, h=-1, w=-1):
        image = utils.convert2float(image)
        if crop and (h >= 0 or w >=0):
            image = tf.image.crop_to_bounding_box(image, h, w, self.crop_size, self.crop_size)
        elif crop:
            image = tf.random_crop(image, [self.crop_size, self.crop_size, 3], seed=seed)
        return image

    def pick(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string),
                    'image/gt_file_name': tf.FixedLenFeature([], tf.string),
                    'image/gt_encoded_image': tf.FixedLenFeature([], tf.string)
                })

            image_buffer = features['image/encoded_image']
            gt_image_buffer = features['image/gt_encoded_image']
            aa = features['image/file_name']
            bb = features['image/gt_file_name']

            image = tf.image.decode_png(image_buffer, channels=3)
            gt_image = tf.image.decode_png(gt_image_buffer, channels=3)
            return image, gt_image

    def crop(self):
        image = tf.image.crop_to_bounding_box(self.im1p, self.height, self.width, self.crop_size, self.crop_size)
        gt_image = tf.image.crop_to_bounding_box(self.im2p, self.height, self.width, self.crop_size, self.crop_size)

        image = self.b_preprocess(image, crop=False)
        gt_image = self.b_preprocess(gt_image, crop=False)

        return image, gt_image

    def feed_old(self, seed):
        """
        Args:
            seed: random seed
        Returns:
          images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string),
                    'image/gt_file_name': tf.FixedLenFeature([], tf.string),
                    'image/gt_encoded_image': tf.FixedLenFeature([], tf.string)
                })

            image_buffer = features['image/encoded_image']
            gt_image_buffer = features['image/gt_encoded_image']
            aa = features['image/file_name']
            bb = features['image/gt_file_name']

            image = tf.image.decode_png(image_buffer, channels=3)
            gt_image = tf.image.decode_png(gt_image_buffer, channels=3)
            image = self.b_preprocess(image, seed=seed)
            gt_image = self.b_preprocess(gt_image, seed=seed)

            images, gt_images, aaa, bbb = tf.train.shuffle_batch(
                [image, gt_image, aa, bb], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.batch_size,#self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=0,#self.min_queue_examples,
            )

            # tf.summary.image('_input', images)
        return images, gt_images, aaa, bbb

    def feed(self, seed, val=False):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string),
                    'image/gt_file_name': tf.FixedLenFeature([], tf.string),
                    'image/gt_encoded_image': tf.FixedLenFeature([], tf.string)
                })

            image_buffer = features['image/encoded_image']
            gt_image_buffer = features['image/gt_encoded_image']
            aa = features['image/file_name']
            bb = features['image/gt_file_name']

            image = tf.image.decode_png(image_buffer, channels=3)
            gt_image = tf.image.decode_png(gt_image_buffer, channels=3)

            h = -1
            w = -1
            if val:
                h = random.randint(0, H - self.crop_size)
                w = random.randint(0, W - self.crop_size)
            image = self.b_preprocess(image, seed=seed, h=h, w=w)
            gt_image = self.b_preprocess(gt_image, seed=seed, h=h, w=w)

            images, gt_images, aaa, bbb = tf.train.shuffle_batch(
                [image, gt_image, aa, bb], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.batch_size,  # self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue= 0 #self.min_queue_examples,
            )

            # tf.summary.image('_input', images)
        return images, gt_images, aa, bb

    def pair_feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string),
                    'image/gt_file_name': tf.FixedLenFeature([], tf.string),
                    'image/gt_encoded_image': tf.FixedLenFeature([], tf.string)
                })

            image_buffer = features['image/encoded_image']
            aa = features['image/file_name']

            image = tf.image.decode_png(image_buffer, channels=3)

            image = utils.convert2float(image)
            y = tf.random_crop(image, [self.crop_size*self.scale, self.crop_size*self.scale, 3])
            x = tf.image.resize_images(y, [self.crop_size, self.crop_size], tf.image.ResizeMethod.BILINEAR)

            x_s, y_s, aaa = tf.train.shuffle_batch(
                [x, y, aa], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.batch_size,  # self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=0,  # self.min_queue_examples,
            )

            # tf.summary.image('_input', images)
        return x_s, y_s, aaa


def test_pick():
    TRAIN_FILE_1 = 'data/tfrecords/validation_x.tfrecords'
    with tf.Graph().as_default():
        reader1 = Reader(TRAIN_FILE_1, batch_size=3)

        im, gt = reader1.pick()
        im2, gt2 = reader1.crop()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():

                im_v, gt_v = sess.run([im, gt])
                h, w, bpp = np.shape(im_v)
                h = random.randint(0, h-32)
                w = random.randint(0, w-32)

                im_vv, gt_vv = sess.run([im2, gt2],
                                      feed_dict={reader1.im1p: im_v,
                                                 reader1.im2p: gt_v,
                                                 reader1.height: h,
                                                 reader1.width: w})
                '''
                plt.figure(1)
                plt.subplot(211)
                plt.imshow(im_vv)

                plt.subplot(212)
                plt.imshow(gt_vv)


                plt.show()
                '''
                step += 1

        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def test_reader():
    TRAIN_FILE_1 = '../../data/tfrecords/train_z.tfrecords'

    with tf.Graph().as_default():
        reader1 = Reader(TRAIN_FILE_1, batch_size=3, crop_size=128)

        images, gt, a, b = reader1.feed(seed=200, val=True)
        imgint = utils.batch_convert2int(images)
        gtint = utils.batch_convert2int(gt)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                batch_img, batch_gt, aa, bb = sess.run([imgint, gtint, a, b])
                #print(aa)
                #print(bb)
                i0 = cv2.vconcat([batch_img[0], batch_gt[0]])
                i1 = cv2.vconcat([batch_img[1], batch_gt[1]])
                i2 = cv2.vconcat([batch_img[2], batch_gt[2]])
                i = cv2.hconcat([i0, i1, i2])

                w = cv2.namedWindow('a2', cv2.WINDOW_NORMAL)
                cv2.imshow('a2', i)
                cv2.waitKey(0)
                '''
                plt.figure(1)
                plt.subplot(231)
                plt.imshow(batch_img[0])

                plt.subplot(232)
                plt.imshow(batch_img[1])

                plt.subplot(233)
                plt.imshow(batch_img[2])

                plt.subplot(234)
                plt.imshow(batch_gt[0])

                plt.subplot(235)
                plt.imshow(batch_gt[1])

                plt.subplot(236)
                plt.imshow(batch_gt[2])

                plt.show()
                step += 1
                '''
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()
