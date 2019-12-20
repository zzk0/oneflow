import oneflow as flow
import numpy as np
import math
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from ops import *
import matplotlib.pyplot as plt
from datetime import datetime

# global config

input_size = 28
batch_size = 64
epoch_num = 2

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


def generator(z, trainable=True):
    s_h, s_w = (input_size, input_size)
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    n = z.static_shape[0]
    h0 = linear(z, 64 * 8 * s_h16 * s_w16, trainable=trainable, name="g_linear")
    h0 = flow.reshape(h0, (n, s_h16, s_w16, 64 * 8))
    h0 = batch_norm(h0, trainable=trainable, name="g_bn1") # epsilon check fail?
    h0 = relu(h0)

    h1 = deconv2d(h0, [n, s_h8, s_w8, 64 * 4], trainable=trainable, name="g_deconv1")
    h1 = batch_norm(h1, trainable=trainable, name="g_bn2")
    h1 = relu(h1)

    h2 = deconv2d(h1, [n, s_h4, s_w4, 64 * 2], trainable=trainable, name="g_deconv2")
    h2 = batch_norm(h2, trainable=trainable, name="g_bn3")
    h2 = relu(h2)

    h3 = deconv2d(h2, [n, s_h2, s_w2, 64], trainable=trainable, name="g_deconv3")
    h3 = batch_norm(h3, trainable=trainable, name="g_bn4")
    h3 = relu(h3)

    h4 = deconv2d(h3, [n, s_h, s_w, 1], trainable=trainable, name="g_deconv4")
    h4 = batch_norm(h4, trainable=trainable, name="g_bn5")
    out = tanh(h4)

    return out

def discriminator(image, trainable=True, reuse=False):

    s_h, s_w = (input_size, input_size)
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    h0 = conv2d(image, 64, trainable=trainable, name="d_conv1", reuse=reuse)
    h0 = lrelu(h0)

    h1 = conv2d(h0, 128, trainable=trainable, name="d_conv2", reuse=reuse)
    h1 = batch_norm(h1, trainable=trainable, name="d_bn1", reuse=reuse)
    h1 = lrelu(h1)

    h2 = conv2d(h1, 256, trainable=trainable, name="d_conv3", reuse=reuse)
    h2 = batch_norm(h2, trainable=trainable, name="d_bn2", reuse=reuse)
    h2 = lrelu(h2)

    h3 = conv2d(h2, 512, trainable=trainable, name="d_conv4", reuse=reuse)
    h3 = batch_norm(h3, trainable=trainable, name="d_bn3", reuse=reuse)
    h3 = lrelu(h3)

    h4 = flow.reshape(h3, (-1, s_h16 * s_w16 * 512))
    out = linear(h4, 1, trainable=trainable, name="d_linear", reuse=reuse)

    return out


if __name__ == "__main__":


    @flow.function
    def train_generator(z=flow.input_blob_def((batch_size, 100)),
                        label1=flow.input_blob_def((batch_size, 1))):
        flow.config.train.primary_lr(0.0001)
        flow.config.train.model_update_conf(dict(naive_conf={}))
    
        g_out = generator(z, trainable=True)
        g_logits = discriminator(g_out, trainable=False)
        g_loss = flow.nn.sigmoid_cross_entropy_with_logits(label1, g_logits, name="Gloss_sigmoid_cross_entropy_with_logits")
        g_loss = flow.math.reduce_mean(g_loss)

        flow.losses.add_loss(g_loss)
        return g_loss, g_out
    
    @flow.function
    def train_discriminator(z=flow.input_blob_def((batch_size, 100)),
                            images=flow.input_blob_def((batch_size, 28, 28, 1)),
                            label1=flow.input_blob_def((batch_size, 1)),
                            label0=flow.input_blob_def((batch_size, 1))):
        flow.config.train.primary_lr(0.0001)
        flow.config.train.model_update_conf(dict(naive_conf={}))

        g_out = generator(z, trainable=False)
        g_logits = discriminator(g_out, trainable=True)
        d_loss_fake = flow.nn.sigmoid_cross_entropy_with_logits(label0, g_logits, name="Dloss_fake_sigmoid_cross_entropy_with_logits")

        d_logits = discriminator(images, trainable=True, reuse=True)
        d_loss_real = flow.nn.sigmoid_cross_entropy_with_logits(label1, d_logits, name="Dloss_real_sigmoid_cross_entropy_with_logits")
        d_loss = d_loss_fake + d_loss_real
        d_loss = flow.math.reduce_mean(d_loss)
        flow.losses.add_loss(d_loss)
    
        return d_loss, d_loss_fake, d_loss_real

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float32)
    check_point = flow.train.CheckPoint()
    check_point.init()



    x, _ = load_mnist()
    batch_num = len(x) // batch_size

    for epoch_idx in range(epoch_num):
        for batch_idx in range(batch_num):
            # fake_images = np.random.randn(batch_size, 28, 28, 1).astype(np.float32)

            for j in range(1):
                z = np.random.normal(0, 1, size=(batch_size, 100)).astype(np.float32)
                label1 = np.ones((batch_size, 1)).astype(np.float32)
                label0 = np.zeros((batch_size, 1)).astype(np.float32)
                images = x[batch_idx*batch_size:(batch_idx+1)*batch_size].astype(np.float32)
                d_loss, d_loss_fake, d_loss_real = train_discriminator(z, images, label1, label0).get()

            for j in range(2):
                z = np.random.normal(0,1,size=(batch_size, 100)).astype(np.float32)
                label1 = np.ones((batch_size, 1)).astype(np.float32)
                g_loss, g_out = train_generator(z, label1).get()


            if  (batch_idx + 1) % 10 == 0:
                print(batch_idx + 1, "th batch:")
                print("d_loss:", d_loss.mean())
                print("dloss_real:", d_loss_real.mean())
                print("dloss_fake:", d_loss_fake.mean())
                print("gloss:", g_loss.mean())
            
            if  (batch_idx + 1) % 100 == 0:
                img = np.squeeze(g_out[0]) / 2 + 0.5
                plt.imsave("gout/test_{}_{}.png".format(str(epoch_idx + 1), str(batch_idx + 1)), img)

        # when a epoch finished
        check_point.save("./model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))+ str(epoch_idx))