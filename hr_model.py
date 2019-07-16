import tensorflow as tf
import utils
from generator3 import Generator3
from edsr_t import EDSR
from discriminator2 import Discriminator2
from reader import Reader
import random

REAL_LABEL = 0.9


class ResGAN:
    def __init__(self,
                 Y_train_file='',
                 Z_train_file='',
                 batch_size=16,
                 l1=10,
                 l2=5,
                 l3=2,
                 learning_rate=2e-4,
                 beta1=0.5,
                 beta2=0.999,
                 epsilon=1e-8):
        self.Y_train_file = Y_train_file
        self.Z_train_file = Z_train_file
        self.batch_size = batch_size
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.EDSR = EDSR('edsr', self.is_training)
        self.G3 = Generator3('G3', self.is_training)
        self.D2 = Discriminator2('D2', self.is_training)
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, None, None, 3])  # real x from data
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, None, None, 3])  # fake y from inner GAN
        self.fake_z = tf.placeholder(tf.float32, shape=[self.batch_size, None, None, 3])  # previously generated sample

    def model(self):
        Z_reader = Reader(self.Z_train_file, name='Z', batch_size=self.batch_size, crop_size=128)
        seed = random.seed()
        z, _, _, _ = Z_reader.feed(seed)
        fake_z = self.EDSR(self.y)

        gan_loss = self.generator_adversarial_loss(self.D2, fake_z)
        cyc_loss = self.cycle_consistency_loss(self.G3, fake_z, self.x)
        idt_loss = self.identity_loss(self.EDSR, z)
        ttv_loss = self.total_variation_loss(fake_z)
        dis_loss = self.discriminator_adversarial_loss(self.D2, z, self.fake_z)

        EDSR_loss = gan_loss + cyc_loss + idt_loss + ttv_loss
        G3_loss = cyc_loss
        D2_loss = dis_loss

        # summary
        tf.summary.histogram('D2/true', self.D2(z))
        tf.summary.histogram('D2/fake', self.D2(fake_z))

        tf.summary.scalar('hr_loss/gan', gan_loss)
        tf.summary.scalar('hr_loss/cycle_consistency', cyc_loss)
        tf.summary.scalar('hr_loss/identity', idt_loss)
        tf.summary.scalar('hr_loss/total_variation', ttv_loss)
        tf.summary.scalar('hr_loss/total_loss', EDSR_loss)
        tf.summary.scalar('hr_loss/discriminator_loss', D2_loss)

        # tf.summary.image('Y/y', utils.batch_convert2int(tf.expand_dims(y[0], 0)))
        # tf.summary.image('Z/EDSR_y', utils.batch_convert2int(tf.expand_dims(fake_z[0], 0)))
        # tf.summary.image('G3/G3_z', utils.batch_convert2int(tf.expand_dims(self.G3(z)[0], 0)))

        return EDSR_loss, G3_loss, D2_loss, fake_z

    def generator_adversarial_loss(self, D2, fake_z):
        return tf.reduce_mean(tf.squared_difference(D2(fake_z), REAL_LABEL))

    def cycle_consistency_loss(self, G3, fake_z, x):
        return tf.reduce_mean(tf.squared_difference(G3(fake_z), x)) * self.l1

    def identity_loss(self, EDSR, z):
        #new_shape = tf.slice(tf.shape(z), [1], [2])
        #z_sub = tf.image.resize_bicubic(z, [32, 32], method=tf.ResizeMethodV1.BICUBIC)
        z_sub = tf.image.resize_bicubic(z, [32, 32])
        return tf.reduce_sum(tf.squared_difference(EDSR(z_sub), z)) * self.l2

    def total_variation_loss(self, fake_z):
        dx, dy = tf.image.image_gradients(fake_z)
        return tf.reduce_mean(tf.norm(dx) + tf.norm(dy)) * self.l3

    def discriminator_adversarial_loss(self, D2, z, fake_z):
        error_real = tf.reduce_mean(tf.squared_difference(D2(z), REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(D2(fake_z)))
        loss = (error_real + error_fake) / 2
        return loss

    def optimize(self, EDSR_loss, G3_loss, D2_loss):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            start_decay_step = 40000
            decay_steps = 40000
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

        EDSR_optimizer = make_optimizer(EDSR_loss, self.EDSR.variables, name='AdamEDSR')
        G3_optimizer = make_optimizer(G3_loss, self.G3.variables, name='AdamG3')
        D2_optimizer = make_optimizer(D2_loss, self.D2.variables, name='AdamD2')

        with tf.control_dependencies([EDSR_optimizer, G3_optimizer, D2_optimizer]):
            return tf.no_op(name='optimizers')
