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


# test 1D
def _test(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"],
        )
        v = flow.get_variable(
            name="v",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["S(1)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        x = flow.matmul(x, v)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("x_arr", x_arr.shape, x_arr.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.flatten()))


def _test_train(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"],
        )
        v = flow.get_variable(
            name="v",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["S(1)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        x = flow.matmul(x, v)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(1)"]
        )
        x = flow.math.relu(x)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("x_arr", x_arr.shape, x_arr.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.flatten()))


def _test_gather(test_case, condition):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(
        x: flow.typing.Numpy.Placeholder((1024, 1024)),
        indices: flow.typing.Numpy.Placeholder(shape=(64,), dtype=flow.int32),
    ) -> flow.typing.Numpy:
        if condition == 0:  # (S0, P)->(S0, B)
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            x = flow.gather(x, indices)
            x = flow.hierarchical_parallel_cast(
                x,
                parallel_hierarchy=[2, 2],
                parallel_distribution=["S(0)", "B"],
                name="gather_cast",
            )
        elif condition == 1:  # (P, S1)->(B, S1)
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(1)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
            x = flow.hierarchical_parallel_cast(
                x,
                parallel_hierarchy=[2, 2],
                parallel_distribution=["B", "S(1)"],
                name="gather_cast",
            )
        elif condition == 2:  # (P, S0)->(B, S0)
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"]
            )
            x = flow.gather(x, indices)
            x = flow.hierarchical_parallel_cast(
                x,
                parallel_hierarchy=[2, 2],
                parallel_distribution=["B", "S(0)"],
                name="gather_cast",
            )
        elif condition == 3:  # (S1, P)->(S1, B)
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
            x = flow.hierarchical_parallel_cast(
                x,
                parallel_hierarchy=[2, 2],
                parallel_distribution=["S(1)", "B"],
                name="gather_cast",
            )
        elif condition == 4:  # (P, B)->(B, B)
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
            x = flow.hierarchical_parallel_cast(
                x,
                parallel_hierarchy=[2, 2],
                parallel_distribution=["B", "B"],
                name="gather_cast",
            )
        else:
            pass

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
        # _test(test_case)
        # _test_train(test_case)
        _test_gather(test_case, 0)  # (S0, P)->(S0, B)
        _test_gather(test_case, 1)  # (P, S1)->(B, S1)
        _test_gather(test_case, 2)  # (P, S0)->(B, S0)
        _test_gather(test_case, 3)  # (S1, P)->(S1, B)
        _test_gather(test_case, 4)  # (P, B)->(B, B)


if __name__ == "__main__":
    unittest.main()
