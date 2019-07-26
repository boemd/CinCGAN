import tensorflow as tf
import helpers.utils as utils
from helpers.reader import Reader
from nets.edsr import EDSR
import helpers.ops as ops

REAL_LABEL = 0.9


class E_mod:
    def __init__(self,
                 train_file='',
                 batch_size=2,
                 learning_rate=1e-4,
                 scale=4
                 ):
        self.train_file = train_file
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scale = scale

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.edsr = EDSR(name='edsr', is_training=self.is_training, scale=self.scale,
                         num_blocks=32, feature_size=256, rb_scaling=0.1)

        self.psnr_validation = tf.placeholder(tf.float32, shape=())

        self.val_x = tf.placeholder(tf.uint8, shape=[1, None, None, 3])

    def model(self):
        reader = Reader(self.train_file, name='f', batch_size=self.batch_size, crop_size=48, scale=self.scale)

        x, y, _ = reader.pair_feed()  # float, already cropped
        # x = tf.to_float(utils.batch_convert2int(x))   # 0 255 ma float
        # y = tf.to_float(utils.batch_convert2int(y))  # tf.to_float(

        val_y = self.edsr(tf.to_float(self.val_x))

        fake_y = self.edsr(x)

        loss = tf.reduce_mean(tf.abs(y - fake_y))

        # summary
        mse = tf.reduce_mean(tf.squared_difference(fake_y, y))
        PSNR = tf.constant(255 ** 2, dtype=tf.float32) / mse
        PSNR = tf.constant(10, dtype=tf.float32) * ops.log10(PSNR)

        # Scalar to keep track for loss
        tf.summary.scalar("metrics/loss", loss)
        tf.summary.scalar("metrics/MSE", mse)
        tf.summary.scalar("metrics/PSNR", PSNR)
        tf.summary.scalar('psnr/validation', self.psnr_validation)

        tf.summary.image('EDSR/input', utils.batch_convert2int(tf.expand_dims(x[0], 0)))
        tf.summary.image('EDSR/output', utils.batch_convert2int(tf.expand_dims(fake_y[0], 0)))
        tf.summary.image('EDSR/ground_truth', utils.batch_convert2int(tf.expand_dims(y[0], 0)))

        return loss, val_y

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

        EDSR_optimizer = make_optimizer(loss, self.edsr.variables, name='AdamEDSRSingle')

        with tf.control_dependencies([EDSR_optimizer]):
            return tf.no_op(name='optimizer')

