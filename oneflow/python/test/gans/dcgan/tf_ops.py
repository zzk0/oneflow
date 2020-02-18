import math
import numpy as np
import tensorflow as tf
import os

from tensorflow.python.framework import ops


if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def get_const_initializer():
    return tf.constant_initializer(0.002)

# class batch_norm(object):
#     def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
#         with tf.variable_scope(name):
#             self.epsilon = epsilon
#             self.momentum = momentum
#             self.name = name

#     def __call__(self, x, train=True):
#         return tf.contrib.layers.batch_norm(x,
#                                             decay=self.momentum,
#                                             updates_collections=None,
#                                             epsilon=self.epsilon,
#                                             scale=True,
#                                             is_training=train,
#                                             scope=self.name)

def batch_norm(name="batch_norm"):
    return lambda x : x  #do nothing

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", const_init=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev) if not const_init else get_const_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, d_h, d_w], padding='SAME', data_format='NCHW')

        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases, data_format='NCHW'), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, const_init=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[1], input_.get_shape()[1]],
                            initializer=tf.random_normal_initializer(stddev=stddev) if not const_init else get_const_initializer())

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, 1, d_h, d_w], data_format='NCHW')

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, 1, d_h, d_w], data_format='NCHW')

        biases = tf.get_variable(
            'biases', [output_shape[1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases, data_format='NCHW'), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, const_init=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev) if not const_init else get_const_initializer())
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start) if not const_init else get_const_initializer())
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def load_mnist(data_dir='./data', dataset_name='mnist', transpose=True):
    data_dir = os.path.join(data_dir, dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    # seed = 547
    # np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    if transpose:
        X = np.transpose(X, (0,3,1,2))
        X = np.pad(X, ((0,),(0,),(2,),(2,)), "edge")

    return X/255., y_vec

if __name__ == '__main__':
    x, y = load_mnist()
    print(x.shape)