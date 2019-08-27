from __future__ import print_function
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import os
import shutil
from datetime import datetime
from fmobilefacenet import MobileFacenet
from fmobilefacenet import CreateInitializer

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
config.grpc_use_no_signal()
config.model_load_snapshot_path("")
_MODEL_SAVE = "./model_save-{}".format(
    str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))
config.model_save_snapshots_path(_MODEL_SAVE)
flow.init(config)

_DATA_DIR = "/home/guoran/insightface_proj/ofrecord/train_16_part"
_SINGLE_PIC_DATA_DIR = '/home/guoran/insightface_proj/ofrecord'
_EVAL_DIR = _DATA_DIR
_TRAIN_DIR = _SINGLE_PIC_DATA_DIR


def MobileFacenetDecoder(dl_net, data_dir='', buffer_size=4096):
  return dl_net.DecodeOFRecord(data_dir,
    name='decode', part_name_suffix_length=5,
    random_shuffle_conf={'buffer_size': buffer_size},
    blob=[
    {
      'name': 'encoded',
      'shape': {'dim': [112, 112, 3]},
      'data_type': flow.float,
      'encode_case': {
        'jpeg': {
          'preprocess': [
            {
              'bgr2rgb': {}
            },
            {
              'mirror': {},
            },
          ]
        },
      },
      'preprocess': [{
        'norm_by_channel_conf': {
          'mean_value': [127.5, 127.5, 127.5],
          'std_value': [128.0, 128.0, 128.0],
          'data_format': 'channels_last',
        },
      }, ]
    },
    {
      'name': 'label',
      'shape': {},
      'data_type': flow.int32,
      'encode_case': {'raw': {}},
    },
  ]);

def BuildInsightfaceWithDeprecatedAPI(data_dir):
    dl_net = flow.deprecated.get_cur_job_dlnet_builder()
    decoders = MobileFacenetDecoder(dl_net, data_dir=data_dir, buffer_size = 2048)
    img = decoders['encoded']
    label = decoders['label']
    fc1 = MobileFacenet(img, embedding_size=512)
    fc7 = dl_net.FullyConnected(fc1, name='fc7', units=85742, activation=op_conf_util.kNone,
                                use_bias=True)
    softmax = dl_net.Softmax(fc7,name='softmax')
    softmax_loss = dl_net.SparseCrossEntropy(softmax,label,name = 'softmax_loss')
    return softmax_loss

def TrainInsightface():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(64).data_part_num(1).default_data_type(flow.float)
    job_conf.train_conf()
    job_conf.default_initializer_conf(CreateInitializer('one'))
    job_conf.train_conf().primary_lr = 0.1
    job_conf.train_conf().num_of_batches_in_snapshot = 100
    job_conf.train_conf().model_update_conf.naive_conf.SetInParent()
    job_conf.train_conf().loss_lbn.extend(["softmax_loss/out"])
    return BuildInsightfaceWithDeprecatedAPI(_TRAIN_DIR)


def EvaluateInsightface():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(64).data_part_num(1).default_data_type(flow.float)
    return BuildInsightfaceWithDeprecatedAPI(_EVAL_DIR)


flow.add_job(TrainInsightface)
flow.add_job(EvaluateInsightface)
if __name__ == '__main__':
    with flow.Session() as sess:
        check_point = flow.train.CheckPoint()
        check_point.restore().initialize_or_restore(session=sess)
        fmt_str = '{:>12}  {:>12}  {:>12.3f}'
        print('{:>12}  {:>12}  {:>12}'.format(
            "iter", "loss type", "loss value"))
        for i in range(100):
            print(fmt_str.format(i, "train loss:", sess.run(
                TrainAlexNet).get().mean()))
            if (i + 1) % 10 is 0:
                print(fmt_str.format(i, "eval loss:", sess.run(
                    EvaluateAlexNet).get().mean()))
            if (i + 1) % 100 is 0:
                check_point.save(session=sess)
