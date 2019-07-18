import tensorflow as tf
import utils
from reader import Reader
from discriminator1 import Discriminator1
from discriminator2 import Discriminator2
from generator12 import Generator12
from generator3 import Generator3
from edsr_t import EDSR
import random
import numpy as np

REAL_LABEL = 0.9

class CinCGAN:
    def __init__(self,
                 X_train_file='',
                 Y_train_file='',
                 Z_train_file='',
                 batch_size=2,
                 scale = 4,
                 b1=10,
                 b2=5,
                 b3=0.5,
                 l1=10,
                 l2=5,
                 l3=2,
                 learning_rate=1e-4,
                 beta1=0.5,
                 beta2=0.999,
                 epsilon=1e-8
                 ):

        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.Z_train_file = Z_train_file
        self.batch_size = batch_size
        self.scale = scale
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G1 = Generator12('G1', self.is_training)
        self.G2 = Generator12('G2', self.is_training)
        self.G3 = Generator3('G3', self.is_training)
        self.EDSR = EDSR(name='edsr', is_training=self.is_training, scale=self.scale,
                         num_blocks=32, feature_size=256, rb_scaling=0.1)
        self.D1 = Discriminator1('D1', self.is_training)
        self.D2 = Discriminator2('D2', self.is_training)

        self.fake_y = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3])
        self.fake_z = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3])

        self.psnr_validation_y = tf.placeholder(tf.float32, shape=())
        self.psnr_validation_z = tf.placeholder(tf.float32, shape=())

        self.val_x = tf.placeholder(tf.uint8, shape=[1, None, None, 3])


    def model(self):
        X_reader = Reader(self.X_train_file, name='X', batch_size=self.batch_size)
        Y_reader = Reader(self.Y_train_file, name='Y', batch_size=self.batch_size)
        Z_reader = Reader(self.Z_train_file, name='Z', batch_size=self.batch_size, crop_size=128)

        seed = 1234  # random.seed()
        x, _, _, _ = X_reader.feed(seed)
        y, _, _, _ = Y_reader.feed(seed)
        z, _, _, _ = Z_reader.feed(seed)

        fake_y = self.G1(x)
        fake_z = self.EDSR(fake_y)

        gan_loss_lr = self.generator_adversarial_loss(self.D1, fake_y)
        cyc_loss_lr = self.cycle_consistency_loss(self.G2, fake_y, x) * self.b1
        idt_loss_lr = self.identity_loss(self.G1, y) * self.b2
        ttv_loss_lr = self.total_variation_loss(fake_y) * self.b3
        dis_loss_lr = self.discriminator_adversarial_loss(self.D1, y, self.fake_y)

        gan_loss_hr = self.generator_adversarial_loss(self.D2, fake_z)
        cyc_loss_hr = self.cycle_consistency_loss(self.G3, fake_z, x) * self.l1
        idt_loss_hr = self.new_identity_loss(self.EDSR, z) * self.l2
        ttv_loss_hr = self.total_variation_loss(fake_z) * self.l3
        dis_loss_hr = self.discriminator_adversarial_loss(self.D2, z, self.fake_z)

        G1_loss = (gan_loss_lr + cyc_loss_lr + idt_loss_lr + ttv_loss_lr)
        G2_loss = cyc_loss_lr
        D1_loss = dis_loss_lr

        EDSR_loss = (gan_loss_hr + cyc_loss_hr + idt_loss_hr + ttv_loss_hr)
        G3_loss = cyc_loss_hr
        D2_loss = dis_loss_hr

        v1 = utils.batch_convert2float(self.val_x)
        v2 = self.G1(v1)
        v3 = self.EDSR(v2)
        val_y = utils.batch_convert2int(v2)
        val_z = utils.batch_convert2int(v3)

        # summary
        tf.summary.histogram('D1/true', self.D1(y))
        tf.summary.histogram('D1/fake', self.D1(fake_y))
        tf.summary.histogram('D2/true', self.D2(z))
        tf.summary.histogram('D2/fake', self.D2(fake_z))

        tf.summary.scalar('lr_loss/gan', gan_loss_lr)
        tf.summary.scalar('lr_loss/cycle_consistency', cyc_loss_lr)
        tf.summary.scalar('lr_loss/identity', idt_loss_lr)
        tf.summary.scalar('lr_loss/total_variation', ttv_loss_lr)
        tf.summary.scalar('lr_loss/total_loss', G1_loss)
        tf.summary.scalar('lr_loss/discriminator_loss', D1_loss)

        tf.summary.scalar('hr_loss/gan', gan_loss_hr)
        tf.summary.scalar('hr_loss/cycle_consistency', cyc_loss_hr)
        tf.summary.scalar('hr_loss/identity', idt_loss_hr)
        tf.summary.scalar('hr_loss/total_variation', ttv_loss_hr)
        tf.summary.scalar('hr_loss/total_loss', EDSR_loss)
        tf.summary.scalar('hr_loss/discriminator_loss', D2_loss)

        tf.summary.image('LR/x', utils.batch_convert2int(tf.expand_dims(x[0], 0)))
        tf.summary.image('LR/fake_y', utils.batch_convert2int(tf.expand_dims(fake_y[0], 0)))
        tf.summary.image('HR/fake_z', utils.batch_convert2int(tf.expand_dims(fake_z[0], 0)))

        tf.summary.scalar('psnr/validation_y', self.psnr_validation_y)
        tf.summary.scalar('psnr/validation_z', self.psnr_validation_z)

        return G1_loss, G2_loss, D1_loss, EDSR_loss, G3_loss, D2_loss, val_y, val_z, fake_y, fake_z

    def generator_adversarial_loss(self, D, fake_sample):
        return tf.reduce_mean(tf.squared_difference(D(fake_sample), REAL_LABEL))

    def cycle_consistency_loss(self, G_back, fake_sample, real_sample):
        return tf.reduce_mean(tf.squared_difference(G_back(fake_sample), real_sample))

    def identity_loss(self, G, sample):
        # return tf.reduce_mean(tf.squared_difference(G1(y), y)) * self.b2
        return tf.reduce_mean(tf.abs(G(sample) - sample))

    def identity_sm_loss(self, G1, y, size=1, mean=1.0, std=1.0):
        def smooth(image, size=2, mean=1.0, std=5.0):
            d = tf.distributions.Normal(mean, std)

            vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

            gauss_kernel = tf.einsum('i,j->ij',
                                     vals,
                                     vals)

            kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
            zeros = np.zeros([1 + 2 * size, 1 + 2 * size])
            kernel_ch1 = tf.stack([kernel, zeros, zeros], axis=2)
            kernel_ch2 = tf.stack([zeros, kernel, zeros], axis=2)
            kernel_ch3 = tf.stack([zeros, zeros, kernel], axis=2)
            kernel_3d = tf.stack([kernel_ch1, kernel_ch2, kernel_ch3], axis=3)

            ret = tf.nn.conv2d(image, kernel_3d, strides=[1, 1, 1, 1], padding="SAME")

            return ret

        sy = smooth(y, size, mean, std)
        sg1y = smooth(G1(y), size, mean, std)
        return tf.reduce_mean(tf.abs(sg1y - sy)), sy, sg1y

    def total_variation_loss(self, sample):
        dx, dy = tf.image.image_gradients(sample)
        return tf.reduce_mean(tf.square(dx) + tf.square(dy))
        # return tf.reduce_mean(tf.image.total_variation(fake_y))

    def discriminator_adversarial_loss(self, D, real_sample, fake_sample):
        error_real = tf.reduce_mean(tf.squared_difference(D(real_sample), REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(D(fake_sample)))
        loss = (error_real + error_fake) / 2
        return loss

    def new_identity_loss(self, EDSR, z):
        #new_shape = tf.slice(tf.shape(z), [1], [2])
        #z_sub = tf.image.resize_bicubic(z, [32, 32], method=tf.ResizeMethodV1.BICUBIC)
        z_sub = tf.image.resize_bicubic(z, [32, 32])
        # loss = tf.reduce_sum(tf.squared_difference(EDSR(z_sub), z))
        loss = tf.reduce_mean(tf.squared_difference(EDSR(z_sub), z))
        return loss

    def optimize(self, G1_loss, G2_loss, D1_loss, EDSR_loss, G3_loss, D2_loss):
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
        EDSR_optimizer = make_optimizer(EDSR_loss, self.EDSR.variables, name='AdamEDSR')
        G3_optimizer = make_optimizer(G3_loss, self.G3.variables, name='AdamG3')
        D2_optimizer = make_optimizer(D2_loss, self.D2.variables, name='AdamD2')

        with tf.control_dependencies(
                [G1_optimizer, G2_optimizer, D1_optimizer, EDSR_optimizer, G3_optimizer, D2_optimizer]):
            return tf.no_op(name='optimizers')
