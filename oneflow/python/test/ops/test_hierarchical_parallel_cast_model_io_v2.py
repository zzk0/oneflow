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
import unittest
import numpy as np
import oneflow as flow
from test_util import GenArgList
import unittest
from collections import OrderedDict
from typing import Dict
import os
import shutil


def _test_model_load_v2(test_case, distribution):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io()
    flow.config.enable_model_io_v2(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((12, 4)),) -> flow.typing.Numpy:
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            x = flow.hierarchical_parallel_cast(x, parallel_distribution=distribution,)
            v = flow.get_variable(
                name="v",
                shape=(12, 4),
                parallel_distribution=distribution,
                initializer=flow.ones_initializer(),
            )
            x = x + v
        x = flow.hierarchical_parallel_cast(x, parallel_distribution=["S(1)"])
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    check_point = flow.train.CheckPoint()
    check_point.load("v_model")
    x_arr = np.random.rand(12, 4).astype(np.float32)
    y_arr = test_fn(x_arr)
    np_v = np.fromfile("v_model/v/out", dtype=np.float32)
    np_y = x_arr + np_v.reshape(12, 4)

    print("y_arr (12, 4):", y_arr.shape, y_arr.flatten())
    print("np_y (12, 4):", np_y.shape, np_y.flatten())
    test_case.assertTrue(np.allclose(y_arr, np_y))


def _test_model_init_v2(test_case, distribution):
    print("_test_model_init_v2")
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io()
    flow.config.enable_model_io_v2(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((12, 4)),) -> flow.typing.Numpy:
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            x = flow.hierarchical_parallel_cast(x, parallel_distribution=distribution,)
            v = flow.get_variable(
                name="v",
                shape=(12, 4),
                parallel_distribution=distribution,
                initializer=flow.xavier_uniform_initializer(),
            )
            x = x + v
        x = flow.hierarchical_parallel_cast(x, parallel_distribution=["S(1)"])
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    check_point = flow.train.CheckPoint()
    check_point.init()
    x_arr = np.zeros((12, 4)).astype(np.float32)
    y_arr = test_fn(x_arr)

    print("y_arr (12, 4):", y_arr.shape, y_arr.flatten())


def _test_model_save_v2(test_case, distribution):
    print("_test_model_save_v2")
    if os.path.exists("v2_save"):
        shutil.rmtree("v2_save")
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io()
    flow.config.enable_model_io_v2(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((12, 4)),) -> flow.typing.Numpy:
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            x = flow.hierarchical_parallel_cast(x, parallel_distribution=distribution,)
            v = flow.get_variable(
                name="v",
                shape=(12, 4),
                parallel_distribution=distribution,
                initializer=flow.ones_initializer(),
            )
            x = x + v
        x = flow.hierarchical_parallel_cast(x, parallel_distribution=["S(1)"])
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    check_point = flow.train.CheckPoint()
    check_point.load("v_model")
    check_point.save("v2_save")
    x_arr = np.zeros((12, 4)).astype(np.float32)
    y_arr = test_fn(x_arr)

    np_v = np.fromfile("v_model/v/out", dtype=np.float32)
    np_v_save = np.fromfile("v2_save/v/out", dtype=np.float32)
    print("np_v (12, 4):", np_v.shape, np_v.flatten())
    print("np_v_save (12, 4):", np_v_save.shape, np_v_save.flatten())
    test_case.assertTrue(np.allclose(np_v, np_v_save))


@flow.unittest.skip_unless_1n4d()
class TestHierarchicalParallelCast(flow.unittest.TestCase):
    def test_hierarchy_parallel_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["distribution"] = [
            ["S(0)", "S(0)"],
            ["S(0)", "S(1)"],
            ["S(0)", "B"],
            ["S(1)", "S(0)"],
            ["S(1)", "S(1)"],
            ["S(1)", "B"],
            ["B", "S(0)"],
            ["B", "S(1)"],
            ["B", "B"],
        ]
        for arg in GenArgList(arg_dict):
            print(*arg)
            _test_model_load_v2(test_case, *arg)
            _test_model_init_v2(test_case, *arg)
            _test_model_save_v2(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
