import oneflow as flow
import numpy as np
import torch
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import tensorflow as tf

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float32)


def test_deconv_2d_forward():

    @flow.function
    def ForwardDeconv2dJob(input=flow.input_blob_def((64, 256, 14, 14))):
        weight = flow.get_variable(name="filter", shape=(
            256, 3, 2, 2), dtype=flow.float32, initializer=flow.ones_initializer())
        output = flow.nn.conv2d_transpose(input, weight, strides=2, output_padding=0, 
                                          dilations=1, padding="same", data_format="NCHW"),
        return output

    x = np.random.randn(64, 256, 14, 14).astype(np.float32)
    # oneflow output
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ForwardDeconv2dJob(x).get()
    # torch output
    # deconv = torch.nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=2, stride=1,
    #                                   padding=1, output_padding=0, groups=1, bias=False, dilation=2, padding_mode='zeros')
    # deconv.weight = torch.nn.Parameter(torch.ones([256, 3, 2, 2]), requires_grad=True)
    # torch_out = deconv(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    # tensorflow output
    x = tf.convert_to_tensor(x)
    kernel = tf.ones([256, 3, 2, 2])
    tf_out = tf.nn.conv2d_transpose(x, kernel, output_shape=[64,3,28,28], strides=[1,2,2,1], padding="SAME")
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    tf_out.eval(session=sess)
    print(tf_out)
    if np.allclose(of_out, torch_out, atol=1e-3):
        print("pass forward test!")
    else:
        print("failed")
    print(of_out[0][0][0][0])
    print(of_out[0].shape)
    # print(torch_out[0][0][0])

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
    # print(torch_dout[0][0][0])
    # print(of_dout[0][0][0])
    

if __name__ == "__main__":
    test_deconv_2d_forward()
    # test_deconv_2d_backward()