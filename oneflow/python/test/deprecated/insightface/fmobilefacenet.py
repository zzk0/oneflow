import oneflow.core.operator.op_conf_pb2 as op_conf_pb2


def CreateInitializer(str):
  if str == 'xavier':
    return {'xavier_conf': {'variance_norm': op_conf_pb2.kFanOut,
                            'rnd_type': op_conf_pb2.kNormal,
                            'magnitude': 2}}
  if str == 'zero':
    return {'constant_conf': {'value': 0.0}}
  if str == 'one':
    return {'constant_conf': {'value': 1.0}}

def Conv(input_blob, num_filter=1, kernel=None, stride=None, pad='valid', num_group=1, name=None, suffix='', **kw):
  dl_net = input_blob.dl_net()
  input_blob = dl_net.Conv2D(input_blob, name='%s%s_conv2d' % (name, suffix), filters=num_filter, kernel_size=kernel,
                             data_format='channels_first', strides=stride, padding=pad, group_num=num_group,
                             activation=op_conf_pb2.kNone,
                             use_bias=False, dilation_rate=[1, 1], **kw)
  input_blob = dl_net.Normalization(input_blob, name='%s%s_batchnorm' % (name, suffix), axis=1, momentum=0.9,
                                    epsilon=0.001, scale=True, center=True, activation=op_conf_pb2.kNone)
  #input_blob = dl_net.PRelu(input_blob, name='%s%s_relu' % (name, suffix), data_format='channels_first')
  input_blob = dl_net.Relu(input_blob, name='%s%s_relu' % (name, suffix))

  return input_blob


def Linear(input_blob, num_filter=1, kernel=None, stride=None, pad='valid', num_group=1, name=None, suffix='', **kw):
  dl_net = input_blob.dl_net()
  input_blob = dl_net.Conv2D(input_blob, name='%s%s_conv2d' % (name, suffix), filters=num_filter, kernel_size=kernel,
                             data_format='channels_first', strides=stride, padding=pad, group_num=num_group,
                             activation=op_conf_pb2.kNone,
                             use_bias=False, dilation_rate=[1, 1], **kw)
  input_blob = dl_net.Normalization(input_blob, name='%s%s_batchnorm' % (name, suffix), axis=1, momentum=0.9,
                                    epsilon=0.001, scale=True, center=True, activation=op_conf_pb2.kNone)

  return input_blob


def DResidual_v1(input_blob, num_out=1, kernel=None, stride=None, pad='same', num_group=1, name=None, suffix=''):
  conv = Conv(input_blob=input_blob, num_filter=num_group, kernel=[1, 1], pad='valid', stride=[1, 1],
              name='%s%s_conv_sep' % (name, suffix))
  conv_dw = Conv(input_blob=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride,
                 name='%s%s_conv_dw' % (name, suffix))
  proj = Linear(input_blob=conv_dw, num_filter=num_out, kernel=[1, 1], pad='valid', stride=[1, 1],
                name='%s%s_conv_proj' % (name, suffix))
  return proj


def Residual(input_blob, num_block=1, num_out=1, kernel=None, stride=None, pad='same', num_group=1, name=None,
             suffix=''):
  dl_net = input_blob.dl_net()
  identity = input_blob
  for i in range(num_block):
    shortcut = identity
    conv = DResidual_v1(input_blob=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad,
                        num_group=num_group,
                        name='%s%s_block' % (name, suffix), suffix='%d' % i)
    #identity = dl_net.Concat([conv, shortcut], axis=1)  # in channel axis
    identity = dl_net.Add([conv, shortcut])
  return identity


def MobileFacenet(input_blob, embedding_size=10):
  """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      embedding_size: int, last fully connected layer size.
    Returns:
      logits: the logits outputs of the model.
    """
  dl_net = input_blob.dl_net()

  input_blob = dl_net.Transpose(input_blob, perm=[0, 3, 1, 2])

  conv_1 = Conv(input_blob, num_filter=64, kernel=[3, 3], pad='same', stride=[2, 2], name="conv_1")
  conv_2 = Residual(conv_1, num_block=2, num_out=64, kernel=[3, 3], stride=[1, 1], pad='same', num_group=64,
                    name="res_2")

  conv_23 = DResidual_v1(conv_2, num_out=128, kernel=[3, 3], stride=[2, 2], pad='same', num_group=128,
                         name="dconv_23")
  conv_3 = Residual(conv_23, num_block=6, num_out=128, kernel=[3, 3], stride=[1, 1], pad='same', num_group=128,
                    name="res_3")

  conv_34 = DResidual_v1(conv_3, num_out=256, kernel=[3, 3], stride=[2, 2], pad='same', num_group=256,
                         name="dconv_34")
  conv_4 = Residual(conv_34, num_block=10, num_out=256, kernel=[3, 3], stride=[1, 1], pad='same', num_group=256,
                    name="res_4")

  conv_45 = DResidual_v1(conv_4, num_out=512, kernel=[3, 3], stride=[2, 2], pad='same', num_group=512,
                         name="dconv_45")
  conv_5 = Residual(conv_45, num_block=2, num_out=512, kernel=[3, 3], stride=[1, 1], pad='same', num_group=512,
                    name="res_5")
  conv_6_sep = Conv(conv_5, num_filter=512, kernel=[1, 1], pad='valid', stride=[1, 1], name="conv_6sep")

  conv_6_dw = Linear(conv_6_sep, num_filter=512, num_group=512, kernel=[7, 7], pad='valid', stride=[1, 1],
                     name="conv_6dw7_7")
  # TODO: use_bias ??
  conv_6_f = dl_net.FullyConnected(conv_6_dw, name='pre_fc1', units=embedding_size, activation=op_conf_pb2.kNone,
                                   use_bias=True)
  fc1 = dl_net.Normalization(conv_6_f, name='fc1', axis=1, momentum=0.9,
                             epsilon=2e-5, scale=False, center=True, activation=op_conf_pb2.kNone)
  return fc1
