import tensorflow as tf
import helpers.ops as ops
import helpers.utils as utils


class Generator12:
    def __init__(self, name, is_training):
        self.name = name
        self.reuse = False
        self.is_training = is_training

    def __call__(self, input):
        """
        Args:
          input: batch_size x width x height x 3
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name):
            # 3 convolutional head layers
            c7s1_64 = ops.c7s1_k(input, k=64, is_training=self.is_training,
                                 reuse=self.reuse, name='b_c7s1_64')
            c3s1_64_a = ops.c3s1_k(c7s1_64, k=64, is_training=self.is_training,
                                   reuse=self.reuse, name='b_c3s1_64_a')
            c3s1_64_b = ops.c3s1_k(c3s1_64_a, k=64, is_training=self.is_training,
                                   reuse=self.reuse, name='b_c3s1_64_b')

            # 6 residual blocks
            blocks = ops.n_res_blocks(c3s1_64_b, reuse=self.reuse, n=6)

            # 3 convolutional tail layers
            c3s1_64_c = ops.c3s1_k(blocks, k=64, is_training=self.is_training,
                                   reuse=self.reuse, name='b_c3s1_64_c')
            c3s1_64_d = ops.c3s1_k(c3s1_64_c, k=64, is_training=self.is_training,
                                   reuse=self.reuse, name='b_c3s1_64_d')
            c7s1_3 = ops.c7s1_k(c3s1_64_d, k=3, is_training=self.is_training,
                                reuse=self.reuse, name='b_c7s1_3_b', activation=None)
            # out = tf.clip_by_value(c7s1_3, -1, 1)
            # out2 = input + out poi clip
            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return c7s1_3

    def sample(self, input_img):
        image = tf.clip_by_value(self.__call__(input_img), -1, 1)
        image = utils.batch_convert2int(image)
        image = tf.image.encode_png(tf.squeeze(image, [0]))
        return image

    def sample_f(self, input_img):
        return tf.squeeze(tf.clip_by_value(self.__call__(input_img), -1, 1), [0])

