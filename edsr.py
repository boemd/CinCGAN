import tensorflow as tf
import tensorflow.contrib.slim as slim
from reader import Reader
import utils
import ops


class EDSR:
    def __init__(self, name, is_training=True, scale=4, num_blocks=32, feature_size=256):
        self.name = name
        self.scale = scale
        self.num_blocks = num_blocks
        self.feature_size = feature_size
        self.reuse = False
        #self.is_training = is_training
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    def __call__(self, input):
        with tf.variable_scope(self.name):
            #mean = tf.reduce_mean(input)
            mean = 127.5
            inp = tf.subtract(input, mean)

            x = slim.conv2d(inp, self.feature_size, [3, 3], reuse=self.reuse)

            # Store the output of the first convolution to add later
            conv_1 = x

            scaling_factor = 0.1

            # Add the residual blocks to the model
            for i in range(self.num_blocks):
                x = ops.resBlock(x, self.feature_size, scale=scaling_factor, reuse=self.reuse)

                # One more convolution, and then we add the output of our first conv layer
            x = slim.conv2d(x, self.feature_size, [3, 3], reuse=self.reuse)
            x += conv_1

            # Upsample output of the convolution
            x = ops.upsample(x, self.scale, self.feature_size, None, reuse=self.reuse)

            # One final convolution on the upsampling output
            output = x  # slim.conv2d(x,output_channels,[3,3])
            ret = tf.clip_by_value(output + mean, 0.0, 255.0)
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            return ret

    def sample(self, input_img):
        #image = utils.batch_convert2int(self.__call__(input_img))
        image = tf.image.encode_png(tf.squeeze(self.__call__(input_img), [0]))
        return image

    def model(self, train_file='', batch_size=1):
        reader = Reader(train_file, name='f', batch_size=batch_size, crop_size=48, scale=4)

        x, y, _ = reader.pair_feed()  # float, already cropped
        x = tf.to_float(utils.batch_convert2int(x))
        y = tf.to_float(utils.batch_convert2int(y))

        fake_y = self.__call__(x)

        loss = tf.reduce_mean(tf.abs(y - fake_y))

        # summary
        mse = tf.reduce_mean(tf.squared_difference(fake_y, y))
        PSNR = tf.constant(255 ** 2, dtype=tf.float32) / mse
        PSNR = tf.constant(10, dtype=tf.float32) * ops.log10(PSNR)

        # Scalar to keep track for loss
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("MSE", mse)
        tf.summary.scalar("PSNR", PSNR)

        tf.summary.image('EDSR/input', tf.expand_dims(x[0], 0))
        tf.summary.image('EDSR/output', tf.expand_dims(fake_y[0], 0))
        tf.summary.image('EDSR/ground_truth', tf.expand_dims(y[0], 0))

        return loss

    def optimize(self, loss, starter_learning_rate=1e-4):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            start_decay_step = 200000
            decay_steps = 200000
            decay_rate = 0.5
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            learning_rate = (
                tf.where(
                    condition=tf.greater_equal(global_step, start_decay_step),
                    x=tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step,
                                                 decay_steps=decay_steps, decay_rate=decay_rate, staircase=True),
                    y=starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, name=name)
                        .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        EDSR_optimizer = make_optimizer(loss, self.variables, name='AdamEDSRSingle')

        with tf.control_dependencies([EDSR_optimizer]):
            return tf.no_op(name='optimizer')
