import tensorflow as tf
import ops
import utils


class EDSR:
    def __init__(self, name, is_training, scale=4, num_blocks=2):
        self.name = name
        self.scale = scale
        self.num_blocks = num_blocks
        self.reuse = False
        self.is_training = is_training

    def __call__(self, input):
        with tf.variable_scope(self.name):
            mean = tf.reduce_mean(input)
            input = tf.subtract(input, mean)
            c3s1_64 = ops.c3s1_k(input, k=64, is_training=self.is_training, reuse=self.reuse, name='b_c3s1_64')
            blocks = ops.n_res_blocks(c3s1_64, reuse=self.reuse, n=self.num_blocks)
            sum = tf.add(c3s1_64, blocks)
            ups = ops.upsample(sum, self.scale)
            last = ops.c3s1_k(ups, k=3, is_training=self.is_training, reuse=self.reuse, name='b_c3s1_3')
            out = tf.add(last, mean)
            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return out

    def sample(self, input_img):
        image = utils.batch_convert2int(self.__call__(input_img))
        image = tf.clip_by_value(image, -1, 1)
        image = tf.image.encode_png(tf.squeeze(image, [0]))
        return image

    def sample_f(self, input_img):
        return tf.squeeze(self.__call__(input_img), [0])

