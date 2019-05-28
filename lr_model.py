import tensorflow as tf
import utils
from reader import Reader
from discriminator1 import Discriminator1
from generator12 import Generator12
import random

REAL_LABEL = 0.9


class CleanGAN:
    def __init__(self,
                 X_train_file='',
                 Y_train_file='',
                 batch_size=2,
                 b1=10,
                 b2=5,
                 b3=0.5,
                 learning_rate=2e-4,
                 beta1=0.5,
                 beta2=0.999,
                 epsilon=1e-8
                 ):
        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G1 = Generator12('G1', self.is_training)
        self.G2 = Generator12('G2', self.is_training)
        self.D1 = Discriminator1('D1', self.is_training)
        self.fake_y = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3])
        self.psnr_validation = tf.placeholder(tf.float32, shape=())

        self.val_x = tf.placeholder(tf.uint8, shape=[1, None, None, 3])



    def model(self):
        X_reader = Reader(self.X_train_file, name='X', batch_size=self.batch_size)
        Y_reader = Reader(self.Y_train_file, name='Y', batch_size=self.batch_size)

        seed = random.seed()
        x, x_gt, _, _ = X_reader.feed(seed)
        y, y_gt, _, _ = Y_reader.feed(seed)

        fake_y = self.G1(x)

        gan_loss = self.generator_adversarial_loss(self.D1, fake_y)
        cyc_loss = self.cycle_consistency_loss(self.G2, fake_y, x)
        idt_loss = self.identity_loss(self.G1, y)
        ttv_loss = self.total_variation_loss(fake_y)
        dis_loss = self.discriminator_adversarial_loss(self.D1, y, self.fake_y)

        G1_loss = (gan_loss + cyc_loss + idt_loss + ttv_loss)
        G2_loss = cyc_loss
        D1_loss = dis_loss

        ################################################################################################################

        #val_y = tf.squeeze(utils.batch_convert2int(self.G1(tf.expand_dims(utils.convert2float(self.val_x), 0))), [0])
        v1 = utils.batch_convert2float(self.val_x)
        v2 = self.G1(v1)
        val_y = utils.batch_convert2int(v2)

        # summary
        tf.summary.histogram('D1/true', self.D1(y))
        tf.summary.histogram('D1/fakeQ', self.D1(self.fake_y))
        tf.summary.histogram('D1/fake', self.D1(self.G1(x)))
        tf.summary.histogram('G1/loss', gan_loss)

        tf.summary.scalar('loss/gan', gan_loss)
        tf.summary.scalar('loss/cycle_consistency', cyc_loss)
        tf.summary.scalar('loss/identity', idt_loss)
        tf.summary.scalar('loss/total_variation', ttv_loss)
        tf.summary.scalar('loss/total_loss', G1_loss)
        tf.summary.scalar('loss/discriminator_loss', D1_loss)
        tf.summary.scalar('psnr/validation', self.psnr_validation)

        tf.summary.image('X/x', utils.batch_convert2int(tf.expand_dims(x[0], 0)))
        tf.summary.image('X/G1_fakey', utils.batch_convert2int(tf.expand_dims(self.G2(fake_y)[0], 0)))
        tf.summary.image('Y/G1_x', utils.batch_convert2int(tf.expand_dims(fake_y[0], 0)))

        tf.summary.image('prev/fake_y', utils.batch_convert2int(tf.expand_dims(self.fake_y[0], 0)))

        return G1_loss, G2_loss, D1_loss, fake_y, val_y

    def generator_adversarial_loss(self, D1, fake_y):
        return tf.reduce_mean(tf.squared_difference(D1(fake_y), REAL_LABEL))# * 0.1

    def cycle_consistency_loss(self, G2, fake_y, x):
        return tf.reduce_mean(tf.squared_difference(G2(fake_y), x)) * self.b1

    def identity_loss(self, G1, y):
        return tf.reduce_mean(tf.squared_difference(G1(y), y)) * self.b2
        # return tf.reduce_mean(tf.abs(G1(y) - y)) * self.b2

    def total_variation_loss(self, fake_y):
        dx, dy = tf.image.image_gradients(fake_y)
        return tf.reduce_mean(tf.square(dx) + tf.square(dy)) * self.b3

    def discriminator_adversarial_loss(self, D1, y, fake_y):
        error_real = tf.reduce_mean(tf.squared_difference(D1(y), REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(D1(fake_y)))
        loss = (error_real + error_fake) / 2
        return loss

    def optimize(self, G1_loss, G2_loss, D1_loss):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            start_decay_step = 20000
            decay_steps = 20000
            decay_rate = 0.5
            beta1 = self.beta1
            beta2 = self.beta2
            epsilon = self.epsilon
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

        G1_optimizer = make_optimizer(G1_loss, self.G1.variables, name='AdamG1')
        G2_optimizer = make_optimizer(G2_loss, self.G2.variables, name='AdamG2')
        D1_optimizer = make_optimizer(D1_loss, self.D1.variables, name='AdamD1')

        with tf.control_dependencies([G1_optimizer, G2_optimizer, D1_optimizer]):
            return tf.no_op(name='optimizers')
