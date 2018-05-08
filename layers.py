from keras.layers import *

class GaussianSample(Layer):
    '''
    Gaussian distribution sampling given mean and variance
    The input has two values, the first is the mean and the second is the variance.
    '''
    def __init__(self,batch_size, **kwargs):
        self.batch_size = batch_size
        super(Sample_z, self).__init__(**kwargs)

    def build(self, input_shape):
        mu_shape = input_shape[0]
        self.n_z = mu_shape[1]
        return mu_shape

    def call(self, inputs, **kwargs):
        mu, l_sigma = inputs
        eps = K.random_normal(shape=(self.batch_size, self.n_z), mean=0., stddev=1.)
        return mu + K.exp(l_sigma / 2) * eps

    def compute_output_shape(self, input_shape):
        return input_shape[0]
