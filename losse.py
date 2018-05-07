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
