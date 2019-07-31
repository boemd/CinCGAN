import tensorflow as tf
import random
import numpy as np
import math
import cv2


def convert2int(image):
    """ Transform from float tensor ([-1.,1.]) to int image ([0,255])
    """
    ig = tf.image.convert_image_dtype((image + 1.0) / 2.0, dtype=tf.uint8)

    return ig


def convert2float(image):
    """ Transform from int image ([0,255]) to float tensor ([-1.,1.])
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #return (image / 127.5) - 1.0
    return (image*2)-1


def batch_convert2int(images):
    """
    Args:
      images: 4D float tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D int tensor
    """
    return tf.map_fn(convert2int, images, dtype=tf.uint8)


def batch_convert2float(images):
    """
    Args:
      images: 4D int tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D float tensor
    """
    return tf.map_fn(convert2float, images, dtype=tf.float32)


class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image


def psnr(im1, im2):
    e = im1.astype("double")/255 - im2.astype("double")/255
    n = im1.shape[0] * im1.shape[1] * im1.shape[2]
    return round(10 * math.log10(n / np.sum(np.power(e, 2))), 4)


def ssim(im1, im2):
    h, w, d = im1.shape
    ssim = 0
    for i in range(d):
        a = im1[:, :, i]
        b = im2[:, :, i]
        K = [0.01, 0.03]
        L = 255

        C1 = (K[0]*L)**2
        C2 = (K[1]*L)**2
        a = a.astype(float)
        b = b.astype(float)
        mu1 = cv2.GaussianBlur(a, (11, 11), 1.5, cv2.BORDER_ISOLATED)
        mu1 = mu1[5:, 5:]
        mu1 = mu1[:-5, :-5]
        mu2 = cv2.GaussianBlur(b.astype(float), (11, 11), 1.5, cv2.BORDER_ISOLATED)
        mu2 = mu2[5:, 5:]
        mu2 = mu2[:-5, :-5]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = np.multiply(mu1, mu2)

        sigma1_sq = cv2.GaussianBlur(a**2, (11, 11), 1.5)
        sigma1_sq = sigma1_sq[5:, 5:]
        sigma1_sq = sigma1_sq[:-5, :-5] - mu1_sq

        sigma2_sq = cv2.GaussianBlur(b**2, (11, 11), 1.5)
        sigma2_sq = sigma2_sq[5:, 5:]
        sigma2_sq = sigma2_sq[:-5, :-5] - mu2_sq

        sigma12 = cv2.GaussianBlur(np.multiply(a, b), (11, 11), 1.5)
        sigma12 = sigma12[5:, 5:]
        sigma12 = sigma12[:-5, :-5] - mu1_mu2

        if C1 > 0 and C2 > 0:
            ssim_map = np.divide(np.multiply((2 * mu1_mu2 + C1), (2 * sigma12 + C2)), np.multiply((mu1_sq + mu2_sq + C1), (sigma1_sq + sigma2_sq + C2)))
        else:
            # this is useless
            numerator1 = 2 * mu1_mu2 + C1
            numerator2 = 2 * sigma12 + C2
            denominator1 = mu1_sq + mu2_sq + C1
            denominator2 = sigma1_sq + sigma2_sq + C2
            ssim_map = np.ones((h, w))
            index = np.nonzero(np.clip(np.dot(denominator1, denominator2), a_min=0))
            ssim_map[index] = np.dot(numerator1[index], numerator2[index]) / \
                              np.dot(denominator1[index], denominator2[index])
            index = np.nonzero(denominator1) and np.argwhere(denominator2 == 0)
            ssim_map[index] = numerator1[index] / denominator1[index]
        ssim += np.mean(ssim_map)
    ssim /= d
    return np.round(ssim, 4)
