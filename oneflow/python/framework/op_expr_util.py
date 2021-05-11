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
import line_profiler
import atexit

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

try:
    import config

    has_config = True
except:
    has_config = False


def push(name):
    if not has_config:
        return
    if not config.warming:
        flow.profiler.range_push(name)


def pop():
    if not has_config:
        return
    if not config.warming:
        flow.profiler.range_pop()


# @profile
def user_op_expr_call(self, *args, **kwargs):
    push("user_op_expr_call")
    # push("user_op_expr_call -- python tensor determine")
    args = list(args)
    for i in range(len(args)):
        arg = args[i]
        if isinstance(arg, flow.Tensor):
            if not arg.is_determined:
                arg.determine()
            args[i] = arg._local_or_consistent_tensor
    # pop()

    attrs = oneflow._oneflow_internal.MutableCfgAttrMap()
    for attr_name, attr_value in kwargs.items():
        assert isinstance(attr_name, str)
        attrs[attr_name] = convert_to_user_attr_value(
            self.op_type_name, attr_name, attr_value
        )
    # pop()

    push("user_op_expr_call -- op expr python apply")
    results = self.apply(args, attrs)
    pop()

    pop()
    return results


def RegisterMethod4UserOpExpr():
    oneflow._oneflow_internal.one.UserOpExpr.__call__ = user_op_expr_call
