import oneflow as flow
import numpy as np

def TODO_test_check_out_blob_def(test_case):
  out_tensor_def = flow.OutFixedTensorDef("y", (10,), dtype=flow.float)
  @flow.function()
  def Foo(x=flow.FixedTensorDef((10,), dtype=flow.float), y=out_tensor_def):
    return x
  box = flow.Box()
  Foo(np.ones((10,), dtype=np.float32), box)
  y = box.value()
  assert type(y) == type(out_tensor_def)
  assert y.op_name == out_tensor_def.op_name
  assert y.shape == out_tensor_def.shape
  assert y.batch_axis == out_tensor_def.batch_axis
  assert y.split_axis == out_tensor_def.split_axis
