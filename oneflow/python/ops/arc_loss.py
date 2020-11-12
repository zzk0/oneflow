"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import

import os
from typing import Union, Optional, Sequence

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.module as module_util
import oneflow.python.ops.math_unary_elementwise_ops as math_unary_elementwise_ops
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("arc_loss")
def arc_loss(
    x: remote_blob_util.BlobDef,
    label: remote_blob_util.BlobDef,
    margin: float = 0.5,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    depth = x.shape[1]
    y, sin_theta_data = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("ArcLoss_")
        )
        .Op("additive_angular_margin")
        .Input("x", [x])
        .Input("label", [label])
        .Output("y")
        .Output("sin_theta_data")
        .Attr("margin", float(margin))
        .Attr("depth", int(depth))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return y
