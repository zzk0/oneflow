import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import os
import numpy as np
from PIL import Image

def get_const_initializer():
    # return flow.random_normal_initializer(stddev=0.02)
    return flow.constant_initializer(0.002)

def deconv2d(input, output_shape,
             k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.02,
             name=None, trainable=True, reuse=False, const_init=False):
    assert name is not None
    name_ = name if reuse == False else name + "_reuse"
    # weight : [in_channels, out_channels, height, width]
    weight_shape = (input.static_shape[1], output_shape[1], k_h, k_w)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02) if not const_init else get_const_initializer(),
        trainable=trainable,
    )

    output = flow.nn.conv2d_transpose(input, weight, strides=[d_h, d_w], output_shape=output_shape[2:],
                                      padding="SAME", data_format="NCHW", name=name_)
    
    bias = flow.get_variable(
        name + "-bias",
        shape=(output_shape[1],),
        dtype=input.dtype,
        initializer=flow.constant_initializer(0.0),
        trainable=trainable,
    )
    output = flow.nn.bias_add(output, bias, "NCHW")
    return output


def conv2d(input, output_dim, 
       k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.02,
       name=None, trainable=True, reuse=False, const_init=False):
    assert name is not None
    name_ = name if reuse == False else name + "_reuse"

    weight_shape = (output_dim, input.static_shape[1], k_h, k_w) # (output_dim, k_h, k_w, input.static_shape[3]) if NHWC
    weight = flow.get_variable(name + "-weight",
                            shape=weight_shape,
                            dtype=input.dtype,
                            initializer=flow.random_normal_initializer(stddev=0.02) if not const_init else get_const_initializer(),
                            trainable=trainable,
                            )

    output = flow.nn.conv2d(input, weight, strides=[d_h, d_w], 
                            padding="SAME", data_format="NCHW", name=name_)

    bias = flow.get_variable(
        name + "-bias",
        shape=(output_dim,),
        dtype=input.dtype,
        initializer=flow.constant_initializer(0.0),
        trainable=trainable,
    )
    output = flow.nn.bias_add(output, bias, "NCHW")
    return output

# def batch_norm(input, name=None, trainable=True, reuse=False, const_init=False):
#     assert name is not None
#     name_ = name if reuse == False else name + "_reuse"
#     return flow.layers.batch_normalization(
#         inputs=input,
#         axis=1,
#         momentum=0.997,
#         epsilon=0.00002,
#         center=True,
#         scale=True,
#         trainable=trainable,
#         name=name_,
#     )

def batch_norm(input, name=None, trainable=True, reuse=False, const_init=False):
    # do nothing
    return input


def linear(input, units, name=None, trainable=True, reuse=False, const_init=False):
    assert name is not None
    name_ = name if reuse == False else name + "_reuse"

    in_shape = input.static_shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    inputs = (
        flow.reshape(input, (-1, in_shape[-1])) if in_num_axes > 2 else input
    )

    weight = flow.get_variable(
        name="{}-weight".format(name),
        shape=(units, inputs.static_shape[1]),
        dtype=inputs.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02) if not const_init else get_const_initializer(),
        trainable=trainable,
        model_name="weight",
    )

    out = flow.matmul(
        a=inputs,
        b=weight,
        transpose_b=True,
        name=name_ + "matmul",
    )

    bias = flow.get_variable(
        name="{}-bias".format(name),
        shape=(units,),
        dtype=inputs.dtype,
        initializer=flow.random_normal_initializer() if not const_init else get_const_initializer(),
        trainable=trainable,
        model_name="bias",
    )

    out = flow.nn.bias_add(
        out, bias, name=name_ + "_bias_add"
    )

    out = (
        flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out
    )
    return out

def relu(input):
    return flow.keras.activations.relu(input)

def tanh(input):
    return flow.keras.activations.tanh(input)

def lrelu(input, alpha=0.2):
    return flow.keras.activations.leaky_relu(input, alpha)

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


if __name__ == "__main__":

    # @flow.function
    # def test_deconv2d(input=flow.input_blob_def((5, 3, 3, 4))):
    #     output_shape = [5, 6, 6, 8]
    #     output = deconv2d(input, output_shape=output_shape)
    #     return output

    # @flow.function
    # def test_deconv2d(input=flow.input_blob_def((5, 3, 3, 4))):
    #     output_shape = [5, 6, 6, 8]
    #     output = deconv2d(input, output_shape=output_shape)
    #     return output

    # @flow.function
    # def test_conv2d(input=flow.input_blob_def((5, 6, 6, 8))):
    #     output = conv2d(input, output_dim=4)
    #     return output
    
    # @flow.function
    # def test_lrelu(input=flow.input_blob_def((3,))):
    #     output = lrelu(input)
    #     return output

    @flow.function
    def test_load_images():
        return load_images()

    @flow.function
    def test_sigmoid_cross_entropy_with_logits(logits=flow.input_blob_def((5, 3,)),
                                               labels=flow.input_blob_def((5, 3))):
        # logits = flow.keras.activations.sigmoid(logits)
        of_out = flow.nn.sigmoid_cross_entropy_with_logits(labels, logits)             
        return of_out
    # flow.config.gpu_device_num(1)
    # flow.config.default_data_type(flow.float32)
    # check_point = flow.train.CheckPoint()
    # check_point.init()

    # print(test_deconv2d(np.random.randn(5,3,3,4).astype(np.float32)).get().shape)
    # print(test_conv2d(np.random.randn(5,6,6,8).astype(np.float32)).get().shape)
    # inputs=np.random.randn(3).astype(np.float32)
    # print(inputs)
    # print(test_lrelu(inputs).get())

    # test load mnist
    images, y = load_mnist()
    images = np.squeeze(images)
    import matplotlib.pyplot as plt
    plt.imsave("test.png", images[2]/(2*np.max(abs(images[1])))+0.5)
    # print(x.shape)
    # print(y.shape)

    # test sigmoid
    # logits = np.random.randn(5,3).astype(np.float32)
    # labels = np.zeros((5,3)).astype(np.float32)
    # of_out = test_sigmoid_cross_entropy_with_logits(logits, labels).get()
    # print("of_out:", of_out)
    # import tensorflow as tf
    # logits = tf.convert_to_tensor(logits)
    # labels = tf.convert_to_tensor(labels)
    # tf_out = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    # sess = tf.Session()
    # tf_out = tf_out.eval(session=sess)
    # print("tf_out:", tf_out)
    # print(tf_out-of_out)
