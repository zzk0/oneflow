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
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        v = flow.get_variable(
            name="v",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["B", "B"],
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
    print("y_arr", y_arr.flatten()[0:10])
    print("x_arr", x_arr.sum(1).flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.sum(1).flatten()))


# test 2D axis 1 change
def _test0(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((512, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        x = flow.math.relu(x)
        # (2, 2)[s0,s0]->(2,2)[s0,s1]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(1)"],
        )
        x = flow.math.reduce_sum(x, axis=[1], keepdims=True)
        # (2, 2)[s0,p]->(2,2)[s0,B]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        x = flow.math.relu(x)
        # (2, 2)[s0,B]->(2,2)[s0,S0]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(512, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("x_arr", x_arr.sum(1).flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.sum(1).flatten()))


# test 2D axis 0 change
def _test1(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 512)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"],
        )
        x = flow.math.reduce_sum(x, axis=[1], keepdims=True)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(1024, 512).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("y_arr sum", y_arr.sum())
    print("x_arr", x_arr.sum(1).flatten()[0:10])
    print("x_arr sum", x_arr.sum())

    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.sum(1).flatten()))


# axis 01
def _test01(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 512)),) -> flow.typing.Numpy:
        # 121
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"],
        )
        x = flow.math.relu(x)
        # (2,2)[s0,s0]->(2,2)[s1,s0]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"],
        )
        #x = flow.hierarchical_parallel_cast(
        #    x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"],
        #)
        x = flow.math.relu(x)
        # (2,2)[s1,s0]->(2,2)[B,S1]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(1)"],
        )
        x = flow.math.relu(x)
        # (2,2)[B,S1]->[S0,S0]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"]
        )
        x = flow.math.relu(x)
        # (2,2)[S0,S0]->[4]S0
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(1024, 512).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("y_arr shape", y_arr.shape)
    print("x_arr", x_arr.flatten()[0:10])
    print("x_arr shape", x_arr.shape)
    test_case.assertTrue(np.allclose(y_arr, x_arr))


# axis 01
def _test_hie(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 512)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(1)"],
        )
        # x = flow.math.reduce_sum(x, axis=[1], keepdims=True)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(1)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(1024, 512).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("y_arr sum", y_arr.sum())
    print("x_arr", x_arr.flatten()[0:10])
    print("x_arr sum", x_arr.sum())

    test_case.assertTrue(np.allclose(y_arr, x_arr))


def _test_hie2(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(8)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 512)),) -> flow.typing.Numpy:
        # (8)[s0]->(2,4)[s0,s1]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 4], parallel_distribution=["S(0)", "S(1)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(1)"],
        )
        # x = flow.math.reduce_sum(x, axis=[1], keepdims=True)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(1)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(1024, 512).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("y_arr sum", y_arr.sum())
    print("x_arr", x_arr.flatten()[0:10])
    print("x_arr sum", x_arr.sum())

    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr))



# test 2D axis 1 change
def _testtest(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((512, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"],
        )
        x = flow.math.relu(x)
        # (2, 2)[s0,s0]->(2,2)[s0,s1]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"],
        )
        # (2, 2)[s0,p]->(2,2)[s0,B]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"],
        )
        x = flow.math.relu(x)
        # (2, 2)[s0,B]->(2,2)[s0,S0]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(512, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("x_arr", x_arr.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.flatten()))

@flow.unittest.skip_unless_1n4d()
class TestHierarchicalParallelCast(flow.unittest.TestCase):
    def test_hierarchy_parallel_cast(test_case):
        # _test(test_case)
        # _test0(test_case)
        # _test1(test_case)
        _test01(test_case)
        # _test_hie(test_case)
        #_testtest(test_case)


if __name__ == "__main__":
    unittest.main()
