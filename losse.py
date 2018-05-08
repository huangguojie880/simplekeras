import tensorflow as tf
import keras.backend as K

def focal_mse(r = 1):
    '''
    A focal mse loss function
    :param r: The coefficient is several times
    :return: -
    '''
    def mse(y_true, y_pred):
        return K.mean((tf.abs(y_true - y_pred)**r)*tf.losses.sigmoid_cross_entropy( y_pred,y_true))
    return mse

def Kl_standard(mu,l_sigma):
    '''
    Calculate the distance from the standard normal distribution
    :param mu: Mean variables
    :param l_sigma:Variance variable
    :return:Kl loss
    '''
    return (0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

def Kl_gap(mu1, lv1, mu2, lv2):
    '''
    The distance between 1 and 2 on Kl
    :param mu1:The mean of 1
    :param lv1:The variance of 1
    :param mu2:The mean of 2
    :param lv2:The variance of 2
    :return:Kl loss
    '''
    v1 = tf.exp(lv1)
    v2 = tf.exp(lv2)
    mu_diff_sq = tf.square(mu1 - mu2)
    dimwise_kld = .5 * ((lv2 - lv1) + tf.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)
    return K.mean(dimwise_kld)
