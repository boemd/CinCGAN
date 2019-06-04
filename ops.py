import tensorflow as tf
import tensorflow.contrib.slim as slim

# Layers: follow the naming convention used in the original paper
# Generator layers


def c7s1_k(input, k, reuse=False, activation='leaky', slope=0.2, is_training=True, name='c7s1_k'):
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("weights",
                           shape=[7, 7, input.get_shape()[3], k])

        conv = tf.nn.conv2d(input, weights,
                            strides=[1, 1, 1, 1], padding='SAME')

        if activation == 'relu':
            output = tf.nn.relu(conv)
        elif activation == 'tanh':
            output = tf.nn.tanh(conv)
        elif activation == 'leaky':
            output = _leaky_relu(conv, slope)
        else:
            output = conv

        return output


def c3s1_k(input, k, reuse=False, activation='leaky', slope=0.2, is_training=True, name='c7s1_k'):
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("weights",
                           shape=[3, 3, input.get_shape()[3], k])

        conv = tf.nn.conv2d(input, weights,
                            strides=[1, 1, 1, 1], padding='SAME')

        if activation == 'relu':
            output = tf.nn.relu(conv)
        elif activation == 'tanh':
            output = tf.nn.tanh(conv)
        elif activation == 'leaky':
            output = _leaky_relu(conv, slope)
        else:
            output = conv

        return output


def c3s2_k(input, k, reuse=False, activation='leaky', slope=0.2, is_training=True, name='c7s1_k'):
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("weights",
                           shape=[3, 3, input.get_shape()[3], k])

        conv = tf.nn.conv2d(input, weights,
                            strides=[1, 2, 2, 1], padding='VALID')

        if activation == 'relu':
            output = tf.nn.relu(conv)
        elif activation == 'tanh':
            output = tf.nn.tanh(conv)
        elif activation == 'leaky':
            output = _leaky_relu(conv, slope)
        else:
            output = conv

        return output


def n_res_blocks(input, reuse, is_training=True, n=6):
    depth = input.get_shape()[3]
    for i in range(1,n+1):
        output = Rk(input, depth, reuse, is_training, 'R{}_{}'.format(depth, i))
        input = output
    return output


def resBlock(x, channels=64, kernel_size=[3, 3], scale=1, reuse=True):
    tmp = slim.conv2d(x, channels, kernel_size, activation_fn=None, reuse=reuse)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, channels, kernel_size, activation_fn=None, reuse=reuse)
    tmp *= scale
    return x + tmp


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def Rk(input, k,  reuse=False, is_training=True, name=None, activation='leaky'):
    with tf.variable_scope(name, reuse=reuse):
        out1 = c3s1_k(input, k, reuse, activation=activation, slope=0.2, is_training=is_training, name="c3s1_k_block_a")
        out2 = c3s1_k(out1, k, reuse, activation=activation, slope=0.2, is_training=is_training, name="c3s1_k_block_b")
        output = tf.add(out2, input)
    return output


# Discriminator layers


def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='batch', is_training=True, name=None):
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("weights",
                           shape=[4, 4, input.get_shape()[3], k])

        conv = tf.nn.conv2d(input, weights,
                            strides=[1, stride, stride, 1], padding='SAME')

        normalized = _norm(conv, is_training, norm)
        output = _leaky_relu(normalized, slope)
        return output


def last_conv(input, reuse=False, use_sigmoid=False, name=None):
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("weights",
                           shape=[4, 4, input.get_shape()[3], 1])
        biases = _biases("biases", [1])

        conv = tf.nn.conv2d(input, weights,
                            strides=[1, 1, 1, 1], padding='SAME')
        output = conv + biases
        if use_sigmoid:
            output = tf.sigmoid(output)
        return output


# Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
    """ Helper to create an initialized Variable
    Args:
        name: name of the variable
        shape: list of ints
        mean: mean of a Gaussian
      s tddev: standard deviation of a Gaussian
    Returns:
        A trainable variable
    """
    var = tf.get_variable(
        name, shape,
        initializer=tf.random_normal_initializer(
            mean=mean, stddev=stddev, dtype=tf.float32))
    return var


def _biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(name, shape,
                           initializer=tf.constant_initializer(constant))


def _leaky_relu(input, slope):
    return tf.maximum(slope * input, input)


def _norm(input, is_training, norm='instance'):
    """ Use Instance Normalization or Batch Normalization or None
    """
    if norm == 'instance':
        return _instance_norm(input)
    elif norm == 'batch':
        return _batch_norm(input, is_training)
    else:
        return input


def _batch_norm(input, is_training):
    """ Batch Normalization
    """
    with tf.variable_scope("batch_norm"):
        return tf.contrib.layers.batch_norm(input,
                                            decay=0.9,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=is_training)


def _instance_norm(input):
    """ Instance Normalization
    """
    with tf.variable_scope("instance_norm"):
        depth = input.get_shape()[3]
        scale = _weights("scale", [depth], mean=1.0)
        offset = _biases("offset", [depth])
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def upsample(x, scale=2, features=64, activation=tf.nn.relu, reuse=True):
    assert scale in [2, 3, 4]
    x = slim.conv2d(x, features, [3, 3], activation_fn=activation, reuse=reuse)
    if scale == 2:
        ps_features = 3 * (scale ** 2)
        x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation, reuse=reuse)
        # x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x, 2, color=True)
    elif scale == 3:
        ps_features = 3 * (scale ** 2)
        x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation, reuse=reuse)
        # x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x, 3, color=True)
    elif scale == 4:
        ps_features = 3 * (2 ** 2)
        for i in range(2):
            x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation, reuse=reuse)
            # x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
            x = PS(x, 2, color=True)
    return x


def _phase_shift(I, r):
    """
    Borrowed from https://github.com/tetrachrome/subpixel
    Used for subpixel phase shifting after deconv operations
    """
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False):
    """
    Borrowed from https://github.com/tetrachrome/subpixel
    Used for subpixel phase shifting after deconv operations
    """
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc],3)
    else:
        X = _phase_shift(X, r)
    return X


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)
