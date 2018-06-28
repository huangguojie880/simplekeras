from keras.layers import *
import tensorflow as tf

class SubpixelConv2d(Layer):
    """It is a 2D sub-pixel up-sampling layer, usually be used
    Parameters
    ------------
    scale : int
        The up-scaling ratio, a wrong setting will lead to dimension size error.
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`__
    """
    def __init__(self,  scale=2,**kwargs):
        super(SubpixelConv2d, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        X = tf.depth_to_space(inputs, self.scale)
        return X

    def compute_output_shape(self, input_shape):
        if int(input_shape[-1]) / (self.scale**2) % 1 != 0:
            raise Exception(
                "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"
            )
        n_out_channel = int(int(input_shape[-1]) / (self.scale**2))
        return [input_shape[:-1] , n_out_channel]

