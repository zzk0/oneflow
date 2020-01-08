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
        self.img_height = 32
        self.img_width = 32
        self.channels = 1
        self.z_dim = 100

        self.batch_size = 64
        self.epoch_num = 2
        self.lr = 0.0001

        self.save_interval = 300
        self.log_interval = 20
        self.data_format = "NCHW"

    def test(self, model_dir=None, z=None):
        
        # @flow.function
        # def test_G_out(z=flow.input_blob_def((self.batch_size, 100))):
        #     g_out = self.generator(z, trainable=False, const_init=True)
        #     return g_out
        
        # @flow.function
        # def test_D_out(images=flow.input_blob_def((self.batch_size, 1, self.img_height, self.img_width))):
        #     d_out = self.discriminator(images, trainable=False, const_init=True)
        #     return d_out

        @flow.function
        def train_generator(z=flow.input_blob_def((self.batch_size, 100)),
                            label1=flow.input_blob_def((self.batch_size, 1))):
            flow.config.train.primary_lr(self.lr)
            flow.config.train.model_update_conf(dict(naive_conf={}))
        
            g_out = self.generator(z, trainable=True, const_init=True)
            g_logits = self.discriminator(g_out, trainable=False, const_init=True)
            g_loss = flow.nn.sigmoid_cross_entropy_with_logits(label1, g_logits, name="Gloss_sigmoid_cross_entropy_with_logits")
            g_loss = flow.math.reduce_mean(g_loss)

            flow.losses.add_loss(g_loss)
            return g_loss, g_out
        
        @flow.function
        def train_discriminator(z=flow.input_blob_def((self.batch_size, 100)),
                                images=flow.input_blob_def((self.batch_size, 1, self.img_height, self.img_width)),
                                label1=flow.input_blob_def((self.batch_size, 1)),
                                label0=flow.input_blob_def((self.batch_size, 1))):
            flow.config.train.primary_lr(self.lr)
            flow.config.train.model_update_conf(dict(naive_conf={}))

            g_out = self.generator(z, trainable=False, const_init=True)
            g_logits = self.discriminator(g_out, trainable=True, const_init=True)
            d_loss_fake = flow.nn.sigmoid_cross_entropy_with_logits(label0, g_logits, name="Dloss_fake_sigmoid_cross_entropy_with_logits")

            d_logits = self.discriminator(images, trainable=True, reuse=True, const_init=True)
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
        x, _ = load_mnist()

        # batch_images = x[1*self.batch_size:(1+1)*self.batch_size].astype(np.float32)
        # z = np.ones((self.batch_size, self.z_dim)).astype(np.float32)
        # D = test_D_out(batch_images).get()

        # print(D.shape)
        # print(D)

        for batch_idx in range(10):
            z = np.ones((self.batch_size, self.z_dim)).astype(np.float32)
            label1 = np.ones((self.batch_size, 1)).astype(np.float32)
            label0 = np.zeros((self.batch_size, 1)).astype(np.float32)
            for j in range(1):
                images = x[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].astype(np.float32)
                d_loss, d_loss_fake, d_loss_real = train_discriminator(z, images, label1, label0).get()

            for j in range(2):
                g_loss, g_out = train_generator(z, label1).get()
            
            print("d_loss_fake:", d_loss_fake.mean())
            print("d_loss_real:", d_loss_real.mean())
            print("g_loss:", g_loss.mean())
        

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
                                images=flow.input_blob_def((self.batch_size, 1, self.img_height, self.img_width)),
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

    def generator(self, z, trainable=True, const_init=False):
        s_h, s_w = (self.img_height, self.img_width)
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        n = z.static_shape[0]
        h0 = linear(z, 64 * 8 * s_h16 * s_w16, trainable=trainable, name="g_linear", const_init=const_init)
        h0 = flow.reshape(h0, (n, 64*8, s_h16, s_w16))
        h0 = batch_norm(h0, trainable=trainable, name="g_bn1", const_init=const_init) # epsilon check fail?
        h0 = relu(h0)

        h1 = deconv2d(h0, [n, 64*4, s_h8, s_w8], trainable=trainable, name="g_deconv1", const_init=const_init)
        h1 = batch_norm(h1, trainable=trainable, name="g_bn2", const_init=const_init)
        h1 = relu(h1)

        h2 = deconv2d(h1, [n, 64*2, s_h4, s_w4], trainable=trainable, name="g_deconv2", const_init=const_init)
        h2 = batch_norm(h2, trainable=trainable, name="g_bn3")
        h2 = relu(h2)

        h3 = deconv2d(h2, [n, 64, s_h2, s_w2], trainable=trainable, name="g_deconv3", const_init=const_init)
        h3 = batch_norm(h3, trainable=trainable, name="g_bn4")
        h3 = relu(h3)

        h4 = deconv2d(h3, [n, 1, s_h, s_w], trainable=trainable, name="g_deconv4", const_init=const_init)
        out = tanh(h4)

        return out

    def discriminator(self, images, trainable=True, reuse=False, const_init=False):
        s_h, s_w = (self.img_height, self.img_width)
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        h0 = conv2d(images, 64, trainable=trainable, name="d_conv1", reuse=reuse, const_init=const_init)
        h0 = lrelu(h0)

        h1 = conv2d(h0, 128, trainable=trainable, name="d_conv2", reuse=reuse, const_init=const_init)
        h1 = batch_norm(h1, trainable=trainable, name="d_bn1", reuse=reuse, const_init=const_init)
        h1 = lrelu(h1)

        h2 = conv2d(h1, 256, trainable=trainable, name="d_conv3", reuse=reuse, const_init=const_init)
        h2 = batch_norm(h2, trainable=trainable, name="d_bn2", reuse=reuse, const_init=const_init)
        h2 = lrelu(h2)

        h3 = conv2d(h2, 512, trainable=trainable, name="d_conv4", reuse=reuse, const_init=const_init)
        h3 = batch_norm(h3, trainable=trainable, name="d_bn3", reuse=reuse, const_init=const_init)
        h3 = lrelu(h3)

        h4 = flow.reshape(h3, (-1, s_h16 * s_w16 * 512))
        out = linear(h4, 1, trainable=trainable, name="d_linear", reuse=reuse, const_init=const_init)

        return out
    
       


if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.test()