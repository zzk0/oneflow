import oneflow as flow
import numpy as np

flow.env.grpc_use_no_signal()
flow.config.gpu_device_num(2)

flow.config.default_data_type(flow.float)

def Print(prefix):
    def _print(x):
        print(prefix)
        print(x)
    return _print

@flow.function
def DistributeSplit(x = flow.input_blob_def((2, 5), is_dynamic=False)):
  with flow.device_prior_placement("gpu", "0:0"):
    a = flow.identity(x)
    b = flow.math.relu(x)
  y = flow.debug.distribute_concat([a, b])
  (s, t) = flow.debug.distribute_split(y)
  print(x.shape, y.shape, s.shape, t.shape)
  with flow.device_prior_placement("gpu", "0:0"):
    flow.watch(s, "s")
    flow.watch(t, "t")
  return s, t

index = [-2, -1, 0, 1, 2]
data = []
for i in index: data.append(np.ones((2, 5), dtype=np.float32) * i)
for x in data: print(DistributeSplit(x).get())
