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


def _test_gather_train(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(
        x: flow.typing.Numpy.Placeholder((1024, 4)),
        indices: flow.typing.Numpy.Placeholder(shape=(12,), dtype=flow.int32),
    ) -> flow.typing.Numpy:
        indices = flow.identity(indices)
        x = flow.identity(x)
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
            )
            v = flow.get_variable(
                name="v",
                shape=(1024, 4),
                parallel_hierarchy=(2, 2),
                parallel_distribution=["S(0)", "S(0)"],
                initializer=flow.zeros_initializer(),
            )
            x = x + v
            indices = flow.hierarchical_parallel_cast(
                indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
            )
            x = flow.gather(x, indices)
            x = flow.hierarchical_parallel_cast(
                x,
                parallel_hierarchy=[2, 2],
                parallel_distribution=["B", "S(1)"],
                grad_mode="manual",
                grad_parallel_hierarchy=[2, 2],
                grad_parallel_distribution=["B", "B"],
            )
            x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    x_arr = np.random.rand(1024, 4).astype(np.float32)
    indices = np.random.randint(low=0, high=20, size=(12,))
    checkpoint = flow.train.CheckPoint()
    checkpoint.init()
    y_arr = test_fn(x_arr, indices)
    gather_out = x_arr[indices]
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("gather_out", gather_out.shape, gather_out.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), gather_out.flatten()))


def _test_slice(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.identity(x)
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
            )
            x = flow.slice(x, begin=(None, 1), size=(None, x.shape[1] - 1))
            x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    slice_out = x_arr[:, 1:]
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("slice_out", slice_out.shape, slice_out.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), slice_out.flatten()))


def _test_reshape(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.identity(x)
        with flow.scope.placement("gpu", "0:0-3", (2, 2)):
            x = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
            )
            x = flow.reshape(x, (512, 2048))
            x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    y_out = x_arr.reshape(512, 2048)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("reshape_out", y_out.shape, y_out.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), y_out.flatten()))


@flow.unittest.skip_unless_1n4d()
class TestHierarchicalParallelCast(flow.unittest.TestCase):
    def test_hierarchy_parallel_cast(test_case):
        _test_slice(test_case)
        _test_reshape(test_case)
        _test_gather_train(test_case)


if __name__ == "__main__":
    unittest.main()
