import oneflow as flow
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from ops import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN():

    def __init__(self):
        self.img_height = 28
        self.img_width = 28
        self.channels = 1
        self.z_dim = 100

        self.batch_size = 64
        self.epoch_num = 2
        self.lr = 0.0001

        self.save_interval = 300
        self.log_interval = 20

    def test(self, model_dir):
        @flow.function
        def test(z=flow.input_blob_def((self.batch_size, 100))):
            g_out = self.generator(z, trainable=False)
            return g_out    

    
    def train(self, save=False, model_dir=None, condition=False):
        @flow.function
        def train_generator(z=flow.input_blob_def((self.batch_size, 100)),
                            label1=flow.input_blob_def((self.batch_size, 1))):
            flow.config.train.primary_lr(self.lr)
            flow.config.train.model_update_conf(dict(naive_conf={}))
        
            g_out = self.generator(z, trainable=True)
            g_logits = self.discriminator(g_out, trainable=False)
            g_loss = flow.nn.sigmoid_cross_entropy_with_logits(label1, g_logits, name="Gloss_sigmoid_cross_entropy_with_logits")
            g_loss = flow.math.reduce_mean(g_loss)

            flow.losses.add_loss(g_loss)
            return g_loss, g_out
        
        @flow.function
        def train_discriminator(z=flow.input_blob_def((self.batch_size, 100)),
                                images=flow.input_blob_def((self.batch_size, 28, 28, 1)),
                                label1=flow.input_blob_def((self.batch_size, 1)),
                                label0=flow.input_blob_def((self.batch_size, 1))):
            flow.config.train.primary_lr(self.lr)
            flow.config.train.model_update_conf(dict(naive_conf={}))

            g_out = self.generator(z, trainable=False)
            g_logits = self.discriminator(g_out, trainable=True)
            d_loss_fake = flow.nn.sigmoid_cross_entropy_with_logits(label0, g_logits, name="Dloss_fake_sigmoid_cross_entropy_with_logits")

            d_logits = self.discriminator(images, trainable=True, reuse=True)
            d_loss_real = flow.nn.sigmoid_cross_entropy_with_logits(label1, d_logits, name="Dloss_real_sigmoid_cross_entropy_with_logits")
            d_loss = d_loss_fake + d_loss_real
            d_loss = flow.math.reduce_mean(d_loss)
            flow.losses.add_loss(d_loss)
    
            return d_loss, d_loss_fake, d_loss_real
        
        flow.config.gpu_device_num(1)
        flow.config.default_data_type(flow.float32)
        check_point = flow.train.CheckPoint()

        if model_dir is not None:
            check_point.load(args.model_load_dir)
        else:
            check_point.init()
        
        x, y = self.load_data()
        batch_num = len(x) // self.batch_size

        for epoch_idx in range(self.epoch_num):
            for batch_idx in range(batch_num):

                for j in range(1):
                    z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
                    label1 = np.ones((self.batch_size, 1)).astype(np.float32)
                    label0 = np.zeros((self.batch_size, 1)).astype(np.float32)
                    images = x[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].astype(np.float32)
                    d_loss, d_loss_fake, d_loss_real = train_discriminator(z, images, label1, label0).get()

                for j in range(2):
                    z = np.random.normal(0,1,size=(self.batch_size, self.z_dim)).astype(np.float32)
                    label1 = np.ones((self.batch_size, 1)).astype(np.float32)
                    g_loss, g_out = train_generator(z, label1).get()

                
                batch_total = batch_idx + epoch_idx * batch_num * self.batch_size
                if (batch_idx + 1) % self.log_interval == 0:
                    print("{}th epoch, {}th batch, dloss:{:>12.6f}, gloss:{:>12.6f}".format(epoch_idx+1, batch_idx+1, d_loss.mean(), g_loss.mean()))            
                    img = np.squeeze(g_out[0]) / 2 + 0.5
                    if not os.path.exists("gout"):
                        os.mkdir("gout")
                    plt.imsave("gout/test_{}_{}.png".format(str(epoch_idx + 1), str(batch_idx + 1)), img)

                if save:
                    if not os.path.exists("model_save"):
                        os.mkdir("model_save")
                    if (batch_total + 1) % self.save_interval == 0:
                        check_point.save("model_save/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))+ str(epoch_idx))

    def load_data(self, data_set="mnist"):
        if data_set is "mnist":
            return load_mnist()
        else:
            raise NotImplementedError

    def generator(self, z, trainable=True):
        s_h, s_w = (self.img_height, self.img_width)
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

    def discriminator(self, images, trainable=True, reuse=False):
        s_h, s_w = (self.img_height, self.img_width)
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        h0 = conv2d(images, 64, trainable=trainable, name="d_conv1", reuse=reuse)
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
    dcgan = DCGAN()
    dcgan.train()