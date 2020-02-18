import tensorflow as tf
from tf_ops import *

def gen_z_batch(batch, z_dim):
    z = np.linspace(0, 1, z_dim) 
    for i in range(batch-1):
        b = np.linspace(0, 1,100)
        z = np.vstack([z, b])
    return z

class DCGAN(object):
    def __init__(self, sess, input_height=32, input_width=32, crop=True,
         batch_size=2, output_height=32, output_width=32,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         max_to_keep=1,
         input_fname_pattern='*.jpg', checkpoint_dir='ckpts', sample_dir='samples', out_dir='./out', data_dir='./data'):

        self.sess = sess

        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = 1

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')
            self.g_bn0 = batch_norm(name='g_bn0')
            self.g_bn1 = batch_norm(name='g_bn1')
            self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.max_to_keep = max_to_keep

        self.build_model()


    def build_model(self):

        if self.y_dim:
            self.y = tf.placeholder(
                tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        image_dims = [1, self.input_height, self.input_width]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G, self.gtmp = self.generator(self.z, self.y, const_init=True)
        self.D, self.D_logits, self.dtmp = self.discriminator(inputs, self.y, reuse=False, const_init=True)
        self.D_, self.D_logits_, self.dtmp_ = self.discriminator(self.G, self.y, reuse=True, const_init=True)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D))
        self.d_loss_fake = sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_))
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss = tf.reduce_mean(self.d_loss_real + self.d_loss_fake)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
    
    def test(self):
        x, _ = load_mnist()
        batch_images = x[1*self.batch_size:(1+1)*self.batch_size].astype(np.float32)
        batch_z = gen_z_batch(self.batch_size, self.z_dim).astype(np.float32)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        
        G = self.sess.run([self.G],
                feed_dict={
                    self.z: batch_z,
                })
        
        D = self.sess.run([self.D_logits],
                feed_dict={
                    self.inputs: batch_images,
                })

        print('D shape: ', D[0].shape, 'D out: ', D[0].mean())
        print('G shape: ', G[0].shape, 'G out: ', G[0].mean())

    def train(self):
        d_optim = tf.train.GradientDescentOptimizer(0.05) \
                .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.GradientDescentOptimizer(0.05) \
                .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()


        x, _ = load_mnist()
        for batch_idx in range(3):
            batch_z = gen_z_batch(self.batch_size, self.z_dim).astype(np.float32)
            batch_images = x[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].astype(np.float32)
      

            # Update G network
            _, errG = self.sess.run([g_optim, self.g_loss],
                            feed_dict={
                            self.z: batch_z,
                            # self.y: batch_labels,
                            })
            # Update D network
            _, errD = self.sess.run([d_optim, self.d_loss],
                            feed_dict={
                            self.inputs: batch_images,
                            self.z: batch_z,
                            # self.y: batch_labels,
                            })

            print("{}th: d_loss:{}, g_loss:{}".format(batch_idx, errD.mean(), errG.mean()))

    def discriminator(self, image, y=None, reuse=False, const_init=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv', const_init=const_init))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv', const_init=const_init)))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv', const_init=const_init)))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv', const_init=const_init)))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin', const_init=const_init)

                return tf.nn.sigmoid(h4), h4, h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])      
                h1 = concat([h1, y], 1)
                
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')
                
                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None, const_init=False):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True, const_init=const_init)

                self.h0 = tf.reshape(
                    self.z_, [-1, self.gf_dim * 8, s_h16, s_w16])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, self.gf_dim*4, s_h8, s_w8], name='g_h1', with_w=True, const_init=const_init)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, self.gf_dim*2, s_h4, s_w4], name='g_h2', with_w=True, const_init=const_init)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, self.gf_dim*1, s_h2, s_w2], name='g_h3', with_w=True, const_init=const_init)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, 1, s_h, s_w], name='g_h4', with_w=True, const_init=const_init)

                out = tf.nn.tanh(h4)
                return out, out
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                    [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

if __name__ == "__main__":
    with tf.Session() as sess:
        dcgan = DCGAN(sess)
        dcgan.train()