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


def _test_train(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((12, 4)),) -> flow.typing.Numpy:
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"],
            )
            v = flow.get_variable(
                name="v",
                shape=(12, 4),
                parallel_hierarchy=(2, 2),
                parallel_distribution=["S(1)", "S(0)"],
                initializer=flow.ones_initializer(),
            )
            x = x + v
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(1)"]
        )
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    check_point = flow.train.CheckPoint()
    # check_point.init()
    check_point.load("v_model")
    x_arr = np.random.rand(12, 4).astype(np.float32)
    y_arr = test_fn(x_arr)
    np_v = np.fromfile("v_model/v/out", dtype=np.float32)
    np_y = x_arr + np_v.reshape(12, 4)

    print("y_arr (12, 4):", y_arr.shape, y_arr.flatten())
    print("np_y (12, 4):", np_y.shape, np_y.flatten())
    test_case.assertTrue(np.allclose(y_arr, np_y))


@flow.unittest.skip_unless_1n4d()
class TestHierarchicalParallelCast(flow.unittest.TestCase):
    def test_hierarchy_parallel_cast(test_case):
        _test_train(test_case)


if __name__ == "__main__":
    unittest.main()
