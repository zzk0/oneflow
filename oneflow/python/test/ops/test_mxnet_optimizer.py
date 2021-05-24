import unittest
import numpy as np
from collections import OrderedDict

import oneflow as flow
import mxnet as mx
from test_util import GenArgList


def compare_with_mxnet_lars(
        device_type,
        x_shape,
        momentum_beta,
        epsilon,
        lars_coefficient,
        learning_rate,
        weight_decay,
        train_iters,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def testLars(
            random_mask: flow.typing.Numpy.Placeholder(x_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_normal_initializer(0, 0.001),
                trainable=True,
            )
            loss = flow.math.reduce_mean(x * random_mask)
            flow.optimizer.LARS(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                momentum_beta=momentum_beta,
                epsilon=epsilon,
                lars_coefficient=lars_coefficient,
                weight_decay=weight_decay,
            ).minimize(loss)
            return x

    # generate random number sequences
    random_masks_seq = []
    for i in range(train_iters + 1):
        random_masks_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    # OneFlow
    init_value = None
    for i in range(train_iters + 1):
        x = testLars(random_masks_seq[i])
        if i == 0:
            init_value = np.copy(x)

    # MxNet
    mx_var = mx.nd.array(init_value).astype("float16")
    mx_var.attach_grad()
    optimizer_params = {
        'learning_rate': learning_rate,
        'wd': weight_decay,
        'multi_precision': True,
        'eta': lars_coefficient,
        'eps': epsilon,
    }
    mx_opt = mx.optimizer.create('lars', **optimizer_params)
    state = mx_opt.create_state([0], [mx_var])
    for i in range(train_iters):
        with mx.autograd.record():
            random_mask = mx.nd.array(random_masks_seq[i]).astype("float16")
            loss = mx.nd.mean(mx_var * random_mask)
            loss.backward(retain_graph=True)
        mx_opt.update([0], [mx_var], [mx_var.grad], [state])

    y = mx_var.asnumpy()
    assert np.allclose(x, y, rtol=1e-4, atol=1e-4)


@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_lars(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(1000, 1000)]
        arg_dict["momentum_beta"] = [0.0]
        arg_dict["epsilon"] = [1e-9]
        arg_dict["lars_coefficient"] = [0.001]
        arg_dict["learning_rate"] = [1]
        arg_dict["weight_decay"] = [0.0]
        arg_dict["train_iters"] = [1]
        for arg in GenArgList(arg_dict):
            compare_with_mxnet_lars(*arg)


if __name__ == "__main__":
    unittest.main()
