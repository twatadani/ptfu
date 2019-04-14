''' leakyrelu.py: callableでパラメータ指定可能なleakey ReLU '''

from .activation import ActivationFunction
import tensorflow as tf

class LeakyReLU(ActivationFunction):


    def __init__(self, alpha):
        ''' alpha: inputが負の値の時の傾き '''
        super(LeakyReLU, self).__init__('LeakyReLU: alpha=' + str(alpha))
        self.alpha = alpha
        return

    def __call__(self, input_tensor):
        return tf.nn.leaky_relu(input_tensor, self.alpha)

name = 'leakyrelu'
