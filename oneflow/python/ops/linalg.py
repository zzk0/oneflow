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

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api
from typing import Optional


@oneflow_export("matmul", "linalg.matmul")
def matmul(
    a: oneflow_api.BlobDesc,
    b: oneflow_api.BlobDesc,
    transpose_a: bool = False,
    transpose_b: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator applies matrix multiplication to two Blobs. 

    Args:
        a (oneflow_api.BlobDesc): A Blob
        b (oneflow_api.BlobDesc): A Blob
        transpose_a (bool, optional): Whether to transpose A Blob. Defaults to False.
        transpose_b (bool, optional): Whether to transpose B Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def matmul_Job(A: tp.Numpy.Placeholder((3, 3)),
                    B: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.linalg.matmul(A, B)


        A = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 1]]).astype(np.float32)
        B = np.array([[3, 4, 5],
                    [6, 7, 8],
                    [9, 10, 11]]).astype(np.float32)
        out = matmul_Job(A, B)

        # output [[ 3.  4.  5.]
        #         [15. 17. 19.]
        #         [ 9. 10. 11.]]

    """
    assert len(a.shape) == len(b.shape)
    assert len(a.shape) >= 2
    if name is None:
        name = id_util.UniqueStr("Matmul_")
    if len(a.shape) == 2:
        op = (
            flow.user_op_builder(name)
            .Op("matmul")
            .Input("a", [a])
            .Input("b", [b])
            .Output("out")
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
            .Build()
        )
    else:
        op = (
            flow.user_op_builder(name)
            .Op("batch_matmul")
            .Input("a", [a])
            .Input("b", [b])
            .Output("out")
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
            .Build()
        )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("matmul_biasadd", "linalg.matmul_biasadd")
def matmul_biasadd(
    a: oneflow_api.BlobDesc,
    b: oneflow_api.BlobDesc,
    bias: oneflow_api.BlobDesc,
    transpose_a: bool = False,
    transpose_b: bool = False,
    data_format: Optional[str] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator applies matrix multiplication to two Blobs, and adds a bias to Blob.

    Args:
        a (oneflow_api.BlobDesc): A Blob
        b (oneflow_api.BlobDesc): A Blob
        bias (oneflow_api.BlobDesc): A 1-D `Blob` with size matching the channel dimension of value. And has the same type as value unless value is a quantized type.
        transpose_a (bool, optional): Whether to transpose A Blob. Defaults to False.
        transpose_b (bool, optional): Whether to transpose B Blob. Defaults to False.
        data_format (Optional[str], optional): A string. '`N...C'` or '`NC...'`. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.
        
    Raises:
        ValueError: ValueError if data format is unrecognized, if value has less than two dimensions with '`N..C'`/None data_format or value has less than three dimensions with '`NC..'` data_format, if bias is a vector, or if the size of bias does not match the size of the channel dimension of value.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def matmul_biasadd_Job(A: tp.Numpy.Placeholder((3, 3)),
                               B: tp.Numpy.Placeholder((3, 3)),
                               Bias: tp.Numpy.Placeholder((3,)),
        ) -> tp.Numpy:
            return flow.matmul_add_bias(A, B, Bias)


        A = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 1]]).astype(np.float32)
        B = np.array([[3, 4, 5],
                    [6, 7, 8],
                    [9, 10, 11]]).astype(np.float32)
        Bias = np.array([1, 2, 1])
        out = matmul_biasadd_Job(A, B, Bias)

        # output [[ 4.  6.  6.]
        #         [16. 19. 20.]
        #         [ 10. 12. 12.]]

    """
    assert len(a.shape) == len(b.shape)
    assert len(a.shape) >= 2
    if name is None:
        name = id_util.UniqueStr("MatmulBiasadd_")

    if data_format is None:
        bias_add_axis = 1
    else:
        if data_format.startswith("NC"):
            bias_add_axis = 1
        elif data_format.startswith("N") and data_format.endswith("C"):
            bias_add_axis = len(a.shape) - 1
        else:
            raise ValueError("data_format must be of the form `N...C` or `NC...`")

    if len(a.shape) == 2:
        op = (
            flow.user_op_builder(name)
            .Op("matmul_biasadd")
            .Input("a", [a])
            .Input("b", [b])
            .Input("bias", [bias])
            .Output("out")
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
            .Attr("axis", bias_add_axis)
            .Build()
        )
    else:
      op = (
          flow.user_op_builder(name)
          .Op("batch_matmul_biasadd")
          .Input("a", [a])
          .Input("b", [b])
          .Input("bias", [bias])
          .Output("out")
          .Attr("transpose_a", transpose_a)
          .Attr("transpose_b", transpose_b)
          .Attr("axis", bias_add_axis)
          .Build()
      )
    return op.InferAndTryRun().RemoteBlobList()[0]
