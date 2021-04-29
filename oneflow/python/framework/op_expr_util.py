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
import oneflow as flow
import oneflow._oneflow_internal
from oneflow.python.framework.attr_util import convert_to_user_attr_value

try:
    import config
except:
    pass


def push(name):
    if not config.warming:
        flow.profiler.range_push(name)


def pop():
    if not config.warming:
        flow.profiler.range_pop()


tensor_cache = []

i = 0


def get_tensor():
    global i
    size = 3
    if i < 3:
        tensor_cache.append(flow.Tensor((1,)))
    ret = tensor_cache[i % 3]
    i += 1
    return ret


def user_op_expr_call(self, *args, **kwargs):
    push("full2")
    push("call1")
    args = list(args)
    for i in range(len(args)):
        arg = args[i]
        if isinstance(arg, flow.Tensor):
            if not arg.is_determined:
                arg.determine()
            args[i] = arg._local_or_consistent_tensor
    pop()

    push("call2")
    attrs = oneflow._oneflow_internal.AttrValueMap()
    for attr_name, attr_value in kwargs.items():
        assert isinstance(attr_name, str)
        attrs[attr_name] = convert_to_user_attr_value(
            self.op_type_name, attr_name, attr_value
        )
    pop()

    push("call3")
    results = self.apply(args, attrs)
    pop()
    push("call3.5")
    # results = list(results)
    def new_list(x):
        b = []
        for i in range(len(x)):
            b.append(x[i])
        return b

    new_res = new_list(results)
    pop()

    # if len(results) > 0:
    #     push("call_test")
    #     for _ in range(10):
    #         a = list(results)
    #     pop()
    #     push("call_test2")
    #     for _ in range(10):
    #         b = []
    #         for i in range(len(results)):
    #             b.append(results[i])
    #     pop()

    # push("call_test2")
    # for _ in range(100):
    # new_res = [None] * len(results)
    # pop()

    push("call4")
    for i, out in enumerate(new_res):
        push("call4.1")
        tensor = get_tensor()
        push("call4.2")
        tensor._local_or_consistent_tensor = out
        push("call4.3")
        tensor._undetermined_tensor = None
        push("call4.4")
        new_res[i] = tensor
        pop()
        pop()
        pop()
        pop()
    results = new_res
    pop()
    pop()

    return results


def RegisterMethod4UserOpExpr():
    oneflow._oneflow_internal.one.UserOpExpr.__call__ = user_op_expr_call
