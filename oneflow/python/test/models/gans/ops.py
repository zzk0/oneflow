import oneflow as flow
import oneflow.python.framework.id_util as id_util
import numpy as np
    

def deconv2d(input, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name=None, trainable=True):
    name = name if name is not None else id_util.UniqueStr("Deconv2d_")
    # weight : [in_channels, height, width, out_channels]
    weight_shape = (input.static_shape[3], k_h, k_w, output_shape[3])
    weight = flow.get_variable(
        id_util.UniqueStr(name + "-weight"),
        shape=weight_shape,
        dtype=input.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02),
        trainable=trainable,
    )

    output = flow.nn.conv2d_transpose(input, weight, strides=[d_h, d_w], output_shape=output_shape[-3:-1],
                                      padding="SAME", data_format="NHWC")
    
    bias = flow.get_variable(
        name + "-bias",
        shape=(output_shape[-1],),
        dtype=input.dtype,
        initializer=flow.constant_initializer(0.0),
        trainable=trainable,
    )
    output = flow.nn.bias_add(output, bias, "NHWC")
    return output

def conv2d(input, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name=None, trainable=True):
    name = name if name is not None else id_util.UniqueStr("Conv2d_")
    weight_shape = (output_dim, k_h, k_w, input.static_shape[3])
    weight = flow.get_variable(name + "-weight",
                            shape=weight_shape,
                            dtype=input.dtype,
                            initializer=flow.random_normal_initializer(stddev=0.02),
                            trainable=trainable,
                            )

    output = flow.nn.conv2d(input, weight, strides=[d_h, d_w], 
                            padding="SAME", data_format="NHWC", name=name)

    bias = flow.get_variable(
        name + "-bias",
        shape=(output_dim,),
        dtype=input.dtype,
        initializer=flow.constant_initializer(0.0),
        trainable=trainable,
    )
    output = flow.nn.bias_add(output, bias, "NHWC")
    return output

def batch_norm(input, name=None, trainable=True):
    name = name if name is not None else id_util.UniqueStr("BatchNorm_")
    return flow.layers.batch_normalization(
        inputs=input,
        axis=1,
        momentum=0.9,
        epsilon=1e-4,
        center=True,
        scale=True,
        trainable=trainable,
        name=name,
    )


def linear(input, units, name=None, trainable=True):
    name = name if name is not None else id_util.UniqueStr("BatchNorm_")
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
        initializer=flow.random_normal_initializer(stddev=0.02),
        trainable=trainable,
        model_name="weight",
    )

    out = flow.matmul(
        a=inputs,
        b=weight,
        transpose_b=True,
        name="{}_matmul".format(name),
    )

    bias = flow.get_variable(
        name="{}-bias".format(name),
        shape=(units,),
        dtype=inputs.dtype,
        initializer=flow.random_normal_initializer(),
        trainable=trainable,
        model_name="bias",
    )

    out = flow.nn.bias_add(
        out, bias, name="{}_bias_add".format(name)
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

def load_images(data_dir="/dataset/PNGS/PNG227/of_record_repeated"):
    image_blob_conf = flow.data.BlobConf(
    "encoded",
    shape=(64, 64, 3),
    dtype=flow.float,
    codec=flow.data.ImageCodec([flow.data.ImagePreprocessor("bgr2rgb"),
                                flow.data.ImageResizePreprocessor(64, 64)]),
    preprocessors=[flow.data.NormByChannelPreprocessor(mean_values=(127.5, 127.5, 127.5),
                                                       std_values=(127.5, 127.5, 127.5))],
    )

    label_blob_conf = flow.data.BlobConf(
    "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
    data_dir, (label_blob_conf, image_blob_conf),
    batch_size=8, data_part_num=8, name="decode"
    )

if __name__ == "__main__":

    @flow.function
    def test_deconv2d(input=flow.input_blob_def((5, 3, 3, 4))):
        output = deconv2d(input, output_shape=[5, 6, 6, 8])
        return output

    @flow.function
    def test_conv2d(input=flow.input_blob_def((5, 6, 6, 8))):
        output = conv2d(input, output_dim=4)
        return output
    
    @flow.function
    def test_lrelu(input=flow.input_blob_def((3,))):
        output = lrelu(input)
        return output

    @flow.function
    def test_load_images():
        return load_images()

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float32)
    check_point = flow.train.CheckPoint()
    check_point.init()

    # print(test_deconv2d(np.random.randn(5,3,3,4).astype(np.float32)).get().shape)
    # print(test_conv2d(np.random.randn(5,6,6,8).astype(np.float32)).get().shape)
    # inputs=np.random.randn(3).astype(np.float32)
    # print(inputs)
    # print(test_lrelu(inputs).get())
    labels, images = test_load_images().get()
    import matplotlib.pyplot as plt
    print(images[1][0][:10])
    print(np.max(images[1]))
    plt.imsave("test.png", images[1]/(2*np.max(abs(images[1])))+0.5)
    # print(images.shape)