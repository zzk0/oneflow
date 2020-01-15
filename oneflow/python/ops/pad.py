from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export

import collections

@oneflow_export('pad')
def pad(x, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('Pad_'))
    setattr(op_conf.pad_conf, "in", x.logical_blob_name)
    setattr(op_conf.pad_conf, "out", "out")
    op_conf.pad_conf.pad_left = pad_left
    op_conf.pad_conf.pad_right = pad_right
    op_conf.pad_conf.pad_top = pad_top
    op_conf.pad_conf.pad_bottom = pad_bottom
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
