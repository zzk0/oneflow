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

import os
import random
import unittest
from collections import OrderedDict
from typing import Dict

import numpy as np
from test_util import GenArgList

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp


def _test(test_case, device_num):
    (m, k, n) = (5, 6, 7)
    a_shape = (m, k)
    b_shape = (k, n)
    c_shape = (n,)
    flow.config.gpu_device_num(device_num)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(
        a: flow.typing.Numpy.Placeholder(a_shape),
        b: flow.typing.Numpy.Placeholder(b_shape),
        c: flow.typing.Numpy.Placeholder(c_shape),
    ) -> flow.typing.Numpy:
        var_a = flow.get_variable(
            name="var_a",
            shape=a_shape,
            dtype=flow.float32,
            initializer=flow.ones_initializer(),
            distribute=flow.distribute.split(1),
        )
        a = flow.hierarchical_parallel_cast(a, nd_sbp=["S(1)"])
        a = var_a * a
        out = flow.matmul(a, b)
        out = flow.hierarchical_parallel_cast(out, nd_sbp=["B"])
        c = flow.hierarchical_parallel_cast(c, nd_sbp=["B"])
        out = flow.nn.bias_add(out, c)
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
        flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(out)
        return out

    a = np.random.rand(*a_shape).astype(np.float32)
    b = np.random.rand(*b_shape).astype(np.float32)
    c = np.random.rand(*c_shape).astype(np.float32)
    out = test_fn(a, b, c)
    test_case.assertTrue(np.allclose(out, np.matmul(a, b) + c))


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(
    flow.unittest.env.eager_execution_enabled(), "2-D SBP doesn't work in eager mode"
)
class TestParallelCast(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_on_gpu(test_case):
        _test(test_case, 2)


def _test_gather(test_case, src, dst):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(
        x: flow.typing.Numpy.Placeholder((1024, 1024)),
        indices: flow.typing.Numpy.Placeholder(shape=(64,), dtype=flow.int32),
    ) -> flow.typing.Numpy:
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            if src[0] == "S(0)":
                x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "B"])
                indices = flow.hierarchical_parallel_cast(
                    indices, nd_sbp=["S(0)", "S(0)"]
                )
                if src[1] == "S(0)":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "B"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["S(0)", "S(0)"]
                    )
                elif src[1] == "S(1)":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "S(1)"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["S(0)", "B"]
                    )
                elif src[1] == "P":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "S(0)"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["S(0)", "B"]
                    )
                elif src[1] == "B":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "B"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["S(0)", "B"]
                    )
            elif src[0] == "P":
                x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "S(0)"])
                indices = flow.hierarchical_parallel_cast(indices, nd_sbp=["B", "B"])
                if src[1] == "S(0)":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "B"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["B", "S(0)"]
                    )
                elif src[1] == "S(1)":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "S(1)"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["B", "B"]
                    )
                elif src[1] == "P":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "S(0)"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["B", "B"]
                    )
                elif src[1] == "B":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "B"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["B", "B"]
                    )
            elif src[0] == "B":
                x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "B"])
                indices = flow.hierarchical_parallel_cast(indices, nd_sbp=["B", "B"])
                if src[1] == "S(0)":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "B"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["B", "S(0)"]
                    )
                elif src == "S(1)":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "S(1)"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["B", "B"]
                    )
                elif src == "P":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "S(0)"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["B", "B"]
                    )
                elif src == "B":
                    x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "B"])
                    indices = flow.hierarchical_parallel_cast(
                        indices, nd_sbp=["B", "B"]
                    )
                else:
                    raise NotImplementedError
            x = flow.gather(x, indices)
            x = flow.hierarchical_parallel_cast(x, nd_sbp=dst, name="gather_cast")
            if dst[0] == "S(0)":
                x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "S(0)"])
            elif dst[0] == "B":
                x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "B"])
            elif dst[0] == "S(1)":
                x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(1)", "S(1)"])
            else:
                raise NotImplementedError
        x = flow.hierarchical_parallel_cast(x, nd_sbp=["B"])
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    indices = np.random.randint(low=0, high=1024, size=(64,))
    y_arr = test_fn(x_arr, indices)
    gather_out = x_arr[indices]
    test_case.assertTrue(np.allclose(y_arr.flatten(), gather_out.flatten()))


def _test_train(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io(True)
    flow.config.enable_model_io_v2(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(
        x: flow.typing.Numpy.Placeholder((1024, 4)),
        indices: flow.typing.Numpy.Placeholder(shape=(12,), dtype=flow.int32),
    ) -> flow.typing.Numpy:
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "S(0)"])
            indices = flow.hierarchical_parallel_cast(indices, nd_sbp=["B", "B"])
            x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "B"])
            v = flow.get_variable(
                name="v",
                shape=(1024, 4),
                nd_sbp=["S(0)", "B"],
                initializer=flow.zeros_initializer(),
            )
            x = x + v
            indices = flow.hierarchical_parallel_cast(indices, nd_sbp=["B", "S(0)"])
            x = flow.gather(x, indices)
            x = flow.hierarchical_parallel_cast(
                x, nd_sbp=["B", "S(0)"], grad_mode="manual", grad_nd_sbp=["B", "S(0)"],
            )
            x = flow.math.relu(x)
            x = flow.hierarchical_parallel_cast(x, nd_sbp=["B", "B"])
        x = flow.hierarchical_parallel_cast(x, nd_sbp=["B"])
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.001]), momentum=0
        ).minimize(x)
        return x

    x_arr = np.random.rand(1024, 4).astype(np.float32)
    indices = np.random.randint(low=0, high=20, size=(12,))
    checkpoint = flow.train.CheckPoint()
    checkpoint.init()
    y_arr = test_fn(x_arr, indices)
    gather_out = x_arr[indices]
    test_case.assertTrue(np.allclose(y_arr.flatten(), gather_out.flatten()))


def _test_reshape(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io(True)
    flow.config.enable_model_io_v2(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def FlowJob(x: flow.typing.Numpy.Placeholder((4, 6), dtype=flow.float)):
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            v = flow.get_variable(
                "x",
                shape=(4, 6),
                dtype=flow.float,
                initializer=flow.constant_initializer(0),
                trainable=True,
                nd_sbp=["S(0)", "S(1)"],
            )
            x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "S(1)"])
            x += v
            loss = flow.reshape(x, (4, 2, 3))
        loss = flow.hierarchical_parallel_cast(loss, nd_sbp=["S(0)"])
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0
        ).minimize(loss)
        return loss

    x = np.random.randn(4, 6).astype(np.float32)
    my_loss = FlowJob(x).get()
    test_case.assertTrue(np.allclose(x.flatten(), my_loss.numpy().flatten()))


def _test_reshape_like(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io(True)
    flow.config.enable_model_io_v2(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="predict", function_config=func_config)
    def FlowJob(x: flow.typing.Numpy.Placeholder((4, 3, 2, 3), dtype=flow.float)):
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            v1 = flow.get_variable(
                "v1",
                shape=(4, 3, 2, 3),
                dtype=flow.float,
                initializer=flow.constant_initializer(0),
                trainable=True,
                nd_sbp=["S(0)", "S(2)"],
            )
            v2 = flow.get_variable(
                "v2",
                shape=(4, 3, 6),
                dtype=flow.float,
                initializer=flow.constant_initializer(0),
                trainable=True,
                nd_sbp=["S(0)", "S(2)"],
            )
            x = flow.hierarchical_parallel_cast(x, nd_sbp=["S(0)", "S(2)"])
            x += v1
            loss = flow.reshape_like(x, v2)
        loss = flow.hierarchical_parallel_cast(loss, nd_sbp=["S(0)"])
        return loss

    x = np.random.randn(4, 3, 2, 3).astype(np.float32)
    my_loss = FlowJob(x).get()
    test_case.assertTrue(np.allclose(x.flatten(), my_loss.numpy().flatten()))


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(
    flow.unittest.env.eager_execution_enabled(), "2-D SBP doesn't work in eager mode"
)
class TestHierarchicalParallelCast(flow.unittest.TestCase):
    def test_change_axis1(test_case):
        arg_dict = OrderedDict()
        arg_dict["src"] = [
            ["S(0)", "S(0)"],
            ["S(0)", "S(1)"],
            ["S(0)", "P"],
            ["S(0)", "B"],
        ]
        arg_dict["dst"] = [["S(0)", "S(0)"], ["S(0)", "S(1)"], ["S(0)", "B"]]
        for arg in GenArgList(arg_dict):
            _test_gather(test_case, *arg)

    def test_change_axis0(test_case):
        arg_dict = OrderedDict()
        arg_dict["src"] = [["B", "S(0)"], ["P", "S(0)"]]
        arg_dict["dst"] = [["B", "S(0)"], ["S(1)", "S(0)"]]
        for arg in GenArgList(arg_dict):
            _test_gather(test_case, *arg)

    def test_train(test_case):
        _test_train(test_case)

    def test_reshape(test_case):
        _test_reshape(test_case)

    def test_reshape_like(test_case):
        _test_reshape_like(test_case)


if __name__ == "__main__":
    unittest.main()
