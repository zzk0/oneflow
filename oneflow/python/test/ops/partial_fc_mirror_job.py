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
import oneflow.typing as oft
import numpy as np
import time

flow.config.gpu_device_num(4)
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.indexed_slices_optimizer_conf(
    dict(include_op_names=dict(op_name=["fc7-weight0", "fc7-weight1", "fc7-weight2", "fc7-weight3"]))
)
num_classes = 1000000
emb_size = 128
batch_size = 256
num_sample = 100000
partial_fc = True
indexed_slice_update = True


@flow.global_function(type="train", function_config=func_config)
def PartialFcJob(
    data: oft.Numpy.Placeholder((batch_size, emb_size), dtype=flow.float),
    labels: oft.Numpy.Placeholder((batch_size,), dtype=flow.int32),
):

    labels = labels.with_distribute(flow.distribute.broadcast())
    data_list = flow.advanced.distribute_clone(data)
    fc7_out_list = []
    mapped_label_list = []
    parallel_desc_symbol = flow.current_scope().device_parallel_desc_symbol
    device_tag = parallel_desc_symbol.device_tag
    parallel_id = 0
    for (
        machine_id,
        device_ids,
    ) in parallel_desc_symbol.machine_id2device_id_list.items():
        for device_id in device_ids:
            with flow.scope.placement(
                device_tag, str(machine_id) + ":" + str(device_id)
            ):
                fc7_weight = flow.get_variable(
                    name="fc7-weight" + str(parallel_id),
                    shape=(num_classes, emb_size),
                    dtype=flow.float,
                    initializer=flow.random_normal_initializer(mean=0.0, stddev=0.01),
                    trainable=True,
                    model_name="weight",
                    distribute=flow.distribute.split(0),
                )
                cur_num_sample = num_sample // parallel_desc_symbol.parallel_num
                cur_num_classes = num_classes // parallel_desc_symbol.parallel_num
                cur_class_offset = parallel_id * cur_num_classes
                cur_sample_offset = parallel_id * cur_num_sample
                (mapped_label, sample_idx) = flow.partial_fc_sample(
                    label=labels,
                    num_sample=cur_num_sample,
                    num_classes=cur_num_classes,
                    class_offset=cur_class_offset,
                    sample_offset=cur_sample_offset,
                )
                sampled_weight = flow.gather(params=fc7_weight, indices=sample_idx)
                fc7 = flow.matmul(
                    a=data_list[parallel_id], b=sampled_weight, transpose_b=True
                )
                fc7_out_list.append(fc7)
                mapped_label_list.append(mapped_label)
                parallel_id += 1
    fc7_out = flow.advanced.distribute_concat(fc7_out_list, axis=1)
    fc7_out = fc7_out.with_distribute(flow.distribute.split(1))
    mapped_label_out = flow.advanced.distribute_add(mapped_label_list)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        mapped_label_out, fc7_out, name="softmax_loss"
    )
    flow.optimizer.SGD(
        flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
    ).minimize(loss)
    return loss


# fake labels
labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
data = np.random.rand(batch_size, emb_size).astype(np.float32)

# OneFlow
check_point = flow.train.CheckPoint()
check_point.init()
start_time = time.time()
for i in range(200):
    loss = PartialFcJob(data, labels).get()
time = time.time() - start_time
print("time", time)
print("loss", loss.numpy().mean())
