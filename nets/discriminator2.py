import tensorflow as tf
import helpers.ops as ops


class Discriminator2:
    def __init__(self, name, is_training):
        self.name = name
        self.is_training = is_training
        self.reuse = False

    def __call__(self, input):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """
        with tf.variable_scope(self.name):
            # convolution layers
            C64 = ops.Ck(input, k=64, stride=2, reuse=self.reuse, norm=None,
                         is_training=self.is_training, name='C64')  # (?, w/2, h/2, 64)
            C128 = ops.Ck(C64, k=128, stride=2, reuse=self.reuse, norm='batch',
                          is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
            C256 = ops.Ck(C128, k=256, stride=2, reuse=self.reuse, norm='batch',
                          is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)
            C512 = ops.Ck(C256, k=512, stride=1, reuse=self.reuse, norm='batch',
                          is_training=self.is_training, name='C512')  # (?, w/16, h/16, 512)

            output = ops.last_conv(C512, reuse=self.reuse, name='output')  # (?, w/16, h/16, 1)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output