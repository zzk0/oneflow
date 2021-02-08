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
import oneflow.typing as tp
from test_util import GenArgList
import unittest
from collections import OrderedDict
from typing import Dict
import os
import random

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
        if src == "(S0, S0)": 
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
            )
            x = flow.gather(x, indices)
        elif src == "(S0, S1)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(1)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(S0, P)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(S0, B)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(S1, S0)": 
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"]
            )
            x = flow.gather(x, indices)
        elif src == "(S1, S1)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(S1, P)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(S1, B)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(P, S0)": 
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"]
            )
            x = flow.gather(x, indices)
        elif src == "(P, S1)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(1)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(P, P)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(P, B)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(B, S0)": 
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"]
            )
            x = flow.gather(x, indices)
        elif src == "(B, S1)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(1)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(B, P)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        elif src == "(B, B)":
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
        else:
            raise NotImplementedError
        
        if len(dst) == 2:
            x = flow.hierarchical_parallel_cast(
                x,
                parallel_hierarchy=[2, 2],
                parallel_distribution=dst,
                name="gather_cast",
            )
        elif len(dst) == 1:
            x = flow.hierarchical_parallel_cast(
                x,
                parallel_hierarchy=[4],
                parallel_distribution=dst,
                name="gather_cast",
            )  
        else:
            raise NotImplementedError         
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    indices = np.random.randint(low=0, high=1024, size=(64,))
    y_arr = test_fn(x_arr, indices)
    gather_out = x_arr[indices]
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("gather_out", gather_out.shape, gather_out.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), gather_out.flatten()))


@flow.unittest.skip_unless_1n4d()
class TestHierarchicalParallelCast(flow.unittest.TestCase):
    def test_hierarchy_parallel_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["src"] = ["(S0, S0)", "(S0, S1)", "(S0, P)", "(S0, B)", "(S1, S0)", "(S1, S1)", "(S1, P)", "(S1, B)", "(P, S0)", "(P, S1)", "(P, P)", "(P, B)", "(B, S0)", "(B, S1)", "(B, P)", "(B, B)"]
        arg_dict["dst"] = [["S(0)", "S(0)"], ["S(0)", "S(1)"], ["S(0)", "B"], ["S(1)", "S(0)"], ["S(1)", "S(1)"], ["S(1)", "B"], ["B", "S(0)"], ["B", "S(1)"], ["B", "B"], ["S(0)"], ["S(1)"], ["B"]]
        for arg in GenArgList(arg_dict):
            print(*arg)
            _test_gather(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
