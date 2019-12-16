import oneflow as flow
import numpy as np
import torch
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import tensorflow as tf

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float32)


def test_deconv_2d_forward(dilation=1, padding=2, output_padding=2, stride=3):

    @flow.function
    def ForwardDeconv2dJob(input=flow.input_blob_def((64, 256, 14, 14))):
        weight = flow.get_variable(name="filter", shape=(
            256, 3, 2, 2), dtype=flow.float32, initializer=flow.ones_initializer())
        output = flow.nn.conv2d_transpose(input, weight, strides=stride, output_padding=output_padding, 
                                          dilations=dilation, padding=padding, data_format="NCHW"),
        return output

    x = np.random.randn(64, 256, 14, 14).astype(np.float32)
    # oneflow output
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ForwardDeconv2dJob(x).get()
    # torch output
    deconv = torch.nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=2, stride=stride,
                                      padding=padding, output_padding=output_padding, groups=1, bias=False,
                                      dilation=dilation, padding_mode='zeros')
    deconv.weight = torch.nn.Parameter(torch.ones([256, 3, 2, 2]), requires_grad=True)
    torch_out = deconv(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    if np.allclose(of_out, torch_out, atol=1e-3):
        print("pass forward test!")
    else:
        print("failed")
    # print(of_out[0][0][0][0])
    print(of_out[0].shape)
    # print(torch_out[0][0][0])

def test_deconv_2d_forward_tf(dilation=1, padding='SAME', output_shape=6, stride=2):

    x = np.random.randn(5, 4, 3, 3).astype(np.float32)
    @flow.function
    def ForwardDeconv2dJob(input=flow.input_blob_def((5, 4, 3, 3))):
        weight = flow.get_variable(name="filter", shape=(
            4, 3, 2, 2), dtype=flow.float32, initializer=flow.ones_initializer())
        output = flow.nn.conv2d_transpose(input, weight, strides=stride, output_shape=[output_shape, output_shape], 
                                          dilations=dilation, padding=padding, data_format="NCHW")
        return output

    # oneflow output
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ForwardDeconv2dJob(x).get()
    # tensorflow output
    x = tf.convert_to_tensor(x)
    kernel = tf.ones([2, 2, 3, 4])
    tf_out = tf.nn.conv2d_transpose(x, kernel, output_shape=[5, 3, output_shape, output_shape], 
                                    strides=[1,1,stride,stride], padding=padding, data_format="NCHW")
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    tf_out = tf_out.eval(session=sess)
    print(of_out-tf_out)

    # if np.allclose(of_out, tf_out, atol=1e-3):
    #     print("pass forward test!")
    # else:
    #     print("failed")


def test_deconv_2d_backward():

    @flow.function
    def BackwardDeconv2dJob(input=flow.input_blob_def((64, 3, 28, 28))):
        weight = flow.get_variable(name="filter", shape=(256, 3, 2, 2), 
                                 dtype=flow.float32, initializer=flow.ones_initializer())
        output = flow.nn.conv2d(input, weight, strides=2, padding="valid", data_format="NCHW")
        return output
    
    dy = np.random.randn(64, 3, 28, 28).astype(np.float32)
    # oneflow output
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_dout = BackwardDeconv2dJob(dy).get()
    # torch output
    x = torch.tensor(np.random.randn(64, 256, 14, 14).astype(np.float32),dtype=torch.float32,requires_grad=True)
    deconv = torch.nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=2, stride=2,
                                      padding=0, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
    deconv.weight = torch.nn.Parameter(torch.ones([256, 3, 2, 2]), requires_grad=True)
    torch_out = deconv(x)
    torch_out.backward(torch.from_numpy(dy))
    torch_dout = x.grad.numpy()
    if np.allclose(of_dout, torch_dout, atol=1e-5):
        print("pass backward test!")

    

if __name__ == "__main__":
    test_deconv_2d_forward_tf()
    # test_deconv_2d_backward()