import tensorflow as tf
import helpers.ops as ops
import helpers.utils as utils


class EDSR:
    def __init__(self, name, is_training, scale=4, num_blocks=32, feature_size=256, rb_scaling=0.1):
        self.name = name
        self.scale = scale
        self.num_blocks = num_blocks
        self.feature_size = feature_size
        self.reuse = False
        self.is_training = is_training
        self.rb_scaling = rb_scaling

    def __call__(self, input):
        with tf.variable_scope(self.name):
            #mean = 127.5
            #inp = tf.subtract(input, mean)
            x = ops.c3s1_k(input=input, k=self.feature_size, is_training=self.is_training,
                           reuse=self.reuse, name='edsr_conv_01', activation='')
            conv_1 = x
            for i in range(self.num_blocks):
                x = ops.resBlock_t(x, channels=self.feature_size, kernel_size=3, scale=self.rb_scaling,
                                   reuse=self.reuse, is_training=self.is_training, name='edsr_block_'+str(i))

            x = ops.c3s1_k(input=x, k=self.feature_size, is_training=self.is_training,
                           reuse=self.reuse, name='edsr_conv_02', activation='')

            #x = ops.dec(x+conv_1, k=self.feature_size, reuse=self.reuse, is_training=self.is_training,
                        #scale=self.scale, name='deconv')
            #x = ops.upsample(x+conv_1, self.scale, self.feature_size, None, reuse=self.reuse)
            #x = ops.upsample4(x, features=self.feature_size, activation='relu', reuse=self.reuse, is_training=self.is_training, name='ups4')
            ############################################################################################################
            x = ops.c3s1_k(x, self.feature_size, reuse=self.reuse, activation='relu', is_training=self.is_training, name='ups' + '_conv_init')
            ps_features = 3 * (2 ** 2)
            for i in range(2):
                # x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation, reuse=reuse)
                x = ops.c3s1_k(x, ps_features, reuse=self.reuse, activation='relu', is_training=self.is_training,
                           name='ups' + '_conv' + str(i))
                x = ops.PS(x, 2, color=True)

            ############################################################################################################

            x = ops.c3s1_k(input=x, k=3, is_training=self.is_training,
                           reuse=self.reuse, name='edsr_conv_03', activation='')

            #out = tf.clip_by_value(x + mean, 0.0, 255.0)
            out = tf.clip_by_value(x, -1.0, 1.0)

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return out

    def sample(self, input_img):
        image = self.__call__(tf.to_float(input_img))
        image = tf.image.convert_image_dtype((image+1)/2, dtype=tf.uint8)  #########
        image = tf.image.encode_png(tf.squeeze(image, [0]))
        return image



