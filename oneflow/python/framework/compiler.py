from __future__ import absolute_import

import oneflow.core.job.job_pb2 as job_util
import oneflow.python.lib.core.func_inspect_util as func_inspect_util
import oneflow.python.lib.core.pb_util as pb_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.output_blob_def as output_blob_util
import oneflow.python.ops as ops

from contextlib import contextmanager

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util

def Compile(function_desc, config_proto):
    job_conf = function_desc.job_config_proto
    job_conf.job_name = function_desc.job_func.__name__
    placement_scope = function_desc.function_attribute.default_placement_scope
    if placement_scope is None:
        dev_ids = placement_util.GetDefaultMachineDeviceIds(config_proto.resource)
        placement_scope = placement_util.DevicePriorPlacementScope(*dev_ids)
    distribute_strategy = function_desc.function_attribute.default_distribute_strategy
    if distribute_strategy is None:
        distribute_strategy = distribute_util.DistributeMirroredStrategy()

    compile_context.ResetCurJobContext()
    with _JobBuildAndInferCtx(job_conf.job_name), placement_scope, distribute_strategy:
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)
        _CompileJob(function_desc)
        c_api_util.CurJobBuildAndInferCtx_Complete()

def _CompileJob(function_desc):
    func = function_desc.job_func
    func.__oneflow_arg_blob_defs__ = _GetArgDefault(func)
    inputs = _RecursiveMakeArgBlobs(func.__oneflow_arg_blob_defs__)
    kwarg = dict(allow_cpu_return_op=function_desc.function_attribute.allow_cpu_return_op)
    func.__oneflow_return_remote_blobs__ = _RecursiveMakeRetRemoteBlobs(func(*inputs), kwarg)

@contextmanager
def _JobBuildAndInferCtx(job_name):
    c_api_util.JobBuildAndInferCtx_Open(job_name)
    yield
    c_api_util.JobBuildAndInferCtx_Close()

def _GetArgDefault(func):
    if hasattr(func, '__oneflow_arg_default__'): return func.__oneflow_arg_default__
    return _CloneArgBlobDef(func_inspect_util.GetArgDefaults(func))

def _CloneArgBlobDef(args):
    if isinstance(args, input_blob_util.ArgBlobDef): return args.Clone()
    if isinstance(args, output_blob_util.OutArgBlobDef): return args.Clone()
    if isinstance(args, (tuple, list)): return type(args)(_CloneArgBlobDef(x) for x in args)
    if isinstance(args, dict): return {k: _CloneArgBlobDef(v) for k, v in args}
    raise NotImplementedError("oneflow.function only accepts nested input blob defs")

def _RecursiveMakeArgBlobs(arg_blob_def):
    if isinstance(arg_blob_def, input_blob_util.ArgBlobDef):
        return ops.InputOpByArgBlobDef(arg_blob_def)
    if isinstance(arg_blob_def, output_blob_util.OutArgBlobDef):
        return arg_blob_def
    if isinstance(arg_blob_def, (tuple, list)):
        return type(arg_blob_def)(_RecursiveMakeArgBlobs(x) for x in arg_blob_def)
    if isinstance(arg_blob_def, dict):
        return {k : _RecursiveMakeArgBlobs(v) for k, v in arg_blob_def.items()}
    raise NotImplementedError("oneflow.function accepts "
            + "ArgBlobDefs/OutArgBlobDefs or list/tuple/dict nested ArgBlobDefs/OutArgBlobDefs as argument")

def _RecursiveMakeRetRemoteBlobs(remote_blobs, kwarg):
    if remote_blobs is None: return None
    if isinstance(remote_blobs, remote_blob_util.BlobDef):
        return ops.RetOpByRemoteBlob(remote_blobs, **kwarg)
    if isinstance(remote_blobs, (tuple, list)):
        return type(remote_blobs)(_RecursiveMakeRetRemoteBlobs(x, kwarg) for x in remote_blobs)
    if isinstance(remote_blobs, dict):
        return {k : _RecursiveMakeRetRemoteBlobs(v, kwarg) for k, v in remote_blobs.items()}
    raise NotImplementedError("oneflow.function returns "
            + "RemoteBlob or list/tuple/dict nested RemoteBlob only")
