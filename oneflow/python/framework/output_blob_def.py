from __future__ import absolute_import

import sys

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as lbi_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.placement_context as placement_ctx
from oneflow.python.oneflow_export import oneflow_export
from functools import reduce
import numpy as np
import oneflow

class OutArgBlobDef(blob_desc.BlobDesc):
    def __init__(self, name, shape, dtype, batch_axis, split_axis='same_with_batch_axis'):
        if split_axis == 'same_with_batch_axis': split_axis = batch_axis
        lbi = lbi_util.LogicalBlobId()
        assert name is not None
        lbi.op_name = name
        lbi.blob_name = "out"
        blob_desc.BlobDesc.__init__(self, lbi)
        assert type(shape) is tuple
        for dim in shape:
            assert type(dim) is int
            assert dim > 0
        self.shape_ = shape
        self.dtype_ = dtype
        self.batch_axis_ = batch_axis
        self.split_axis_ = split_axis

    @property
    def static_shape(self): return self.shape_

    @property
    def shape(self): return self.shape_

    @property
    def dtype(self): return self.dtype_

    @property
    def batch_axis(self): return self.batch_axis_

    @property
    def split_axis(self): return self.split_axis_

    @property
    def is_dynamic(self):
        raise NotImplementedError

    @property
    def is_tensor_list(self):
        raise NotImplementedError

    def with_distribute(self, distribute):
        return type(self)(shape = self.shape_,
                          dtype = self.dtype_,
                          batch_axis = self.batch_axis_,
                          name = self.lbi.op_name)
    
    def Clone(self):
        return type(self)(self.op_name, self.shape, dtype=self.dtype,
                          batch_axis = self.batch_axis, split_axis=self.split_axis)

    def AddAndInferOp(self, op_conf):
        raise NotImplementedError

    def ToInterfaceBlobConf(self):
        interface_blob_conf = op_conf_util.InterfaceBlobConf()
        interface_blob_conf.shape.dim.extend(self.shape_)
        interface_blob_conf.data_type = self.dtype_
        interface_blob_conf.is_dynamic = self.is_dynamic
        interface_blob_conf.is_tensor_list = self.is_tensor_list
        if type(self.batch_axis_) is int:
            assert self.batch_axis_ >= 0
            interface_blob_conf.batch_axis.value = self.batch_axis_
        else:
            assert self.batch_axis_ is None or self.batch_axis_ is False
            interface_blob_conf.batch_axis.ClearField("value")
        if type(self.split_axis_) is int:
            assert self.split_axis_ >= 0
            interface_blob_conf.split_axis.value = self.split_axis_
        else:
            assert self.split_axis_ is None or self.split_axis_ is False
            interface_blob_conf.split_axis.ClearField("value")
        return interface_blob_conf

@oneflow_export('OutFixedTensorDef')
class OutFixedTensorDef(OutArgBlobDef):
    def __init__(self, name, shape, dtype=data_type_util.kFloat, batch_axis=0,
                 split_axis='same_with_batch_axis'):
        if type(batch_axis) is int:
            if batch_axis < 0: batch_axis += len(shape)
            assert batch_axis >= 0
            assert batch_axis < len(shape)
        OutArgBlobDef.__init__(self, name, shape, dtype=dtype, batch_axis=batch_axis,
                            split_axis=split_axis)
        
    @property
    def is_dynamic(self): return False

    @property
    def is_tensor_list(self): return False

    def AddAndInferOp(self, op_conf):
        return compile_context.CurJobAddConsistentOp(op_conf)

@oneflow_export('OutMirroredTensorDef')
class OutMirroredTensorDef(OutArgBlobDef):
    def __init__(self, name, shape, dtype=data_type_util.kFloat, batch_axis=0):
        assert type(shape) is tuple
        assert type(batch_axis) is int
        if batch_axis < 0: batch_axis += len(shape)
        assert batch_axis >= 0
        assert batch_axis < len(shape)
        OutArgBlobDef.__init__(self, name, shape, dtype=dtype, batch_axis=batch_axis)
        self.sub_consistent_blob_list_ = []
        
    @property
    def is_dynamic(self): return True

    @property
    def is_tensor_list(self): return False

    def AddAndInferOp(self, op_conf):
        _AddAndInferMirroredOp(self.logical_blob_name, op_conf, self.sub_consistent_blob_list_)
        
@oneflow_export('OutMirroredTensorListDef')
class OutMirroredTensorListDef(OutArgBlobDef):
    def __init__(self, name, shape, dtype=data_type_util.kFloat, batch_axis=0):
        assert type(shape) is tuple
        assert type(batch_axis) is int
        if batch_axis < 0: batch_axis += len(shape)
        assert batch_axis >= 0
        assert batch_axis < len(shape)
        OutArgBlobDef.__init__(self, name, shape, dtype=dtype, batch_axis=batch_axis)
        self.sub_consistent_blob_list_ = []
        
    @property
    def is_dynamic(self): return True

    @property
    def is_tensor_list(self): return True

    def AddAndInferOp(self, op_conf):
        _AddAndInferMirroredOp(self.logical_blob_name, op_conf, self.sub_consistent_blob_list_)
        
def _AddAndInferMirroredOp(mirrored_lbn, op_conf, sub_consistent_blob_list):
    compile_context.CurJobAddMirroredOp(op_conf)
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    num_sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, mirrored_lbn)
    for i in range(num_sub_lbi):
        sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, mirrored_lbn, i)
        sub_consistent_blob_list.append(remote_blob_util.ConsistentBlob(sub_lbi))
