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
import os
import numpy as np
import oneflow as flow
import oneflow.typing as oft
from collections import OrderedDict

from test_util import GenArgList
import test_global_storage
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def compare_with_np(
    device_type, label_type, num_classes, num_sample, batch_size, indexed_slice_update
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.gpu_device_num(0)
        flow.config.cpu_device_num(4)
    else:
        flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(type="train", function_config=func_config)
    def PartialFcJob(
        labels: oft.Numpy.Placeholder(
            (batch_size,), dtype=type_name_to_flow_type[label_type]
        )
    ):
        #labels = labels.with_distribute(flow.distribute.broadcast())
        labels = flow.parallel_cast(labels, distribute=flow.distribute.broadcast())
        labels_list = flow.advanced.distribute_clone(labels)
        mapped_label_list = []
        sampled_weight_list = []
        sampled_label_list = []
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
                    print("label.shape", labels_list[0].shape, labels_list[1].shape, labels_list[2].shape, labels_list[3].shape)
                    cur_num_classes = num_classes // parallel_desc_symbol.parallel_num
                    fc7_weight = flow.get_variable(
                        name="fc7-weight" + str(parallel_id),
                        shape=(cur_num_classes, 128),
                        dtype=flow.float,
                        initializer=flow.random_normal_initializer(mean=0.0, stddev=0.01),
                        trainable=True,
                    )
                    print("fc7_weight",fc7_weight.shape)
                    cur_num_sample = num_sample // parallel_desc_symbol.parallel_num
                    cur_class_offset = parallel_id * cur_num_classes
                    cur_sample_offset = parallel_id * cur_num_sample
                    print("id", parallel_id, cur_class_offset, cur_sample_offset)
                    (mapped_label, sample_idx) = flow.partial_fc_sample(
                        label=labels_list[parallel_id],
                        num_sample=cur_num_sample,
                        num_classes=cur_num_classes,
                        class_offset=cur_class_offset,
                        sample_offset=cur_sample_offset,
                    )
                    print(mapped_label.shape, sample_idx.shape)
                    sampled_weight = flow.gather(params=fc7_weight, indices=sample_idx)
                    sampled_weight_list.append(sampled_weight)
                    mapped_label_list.append(mapped_label)
                    sampled_label_list.append(sample_idx)
                    parallel_id += 1

        sampled_weight_out = flow.advanced.distribute_concat(sampled_weight_list, axis=0)
        sampled_label_out = flow.advanced.distribute_concat(sampled_label_list, axis=0)
        mapped_label_out = flow.advanced.distribute_add(mapped_label_list)

        #with flow.scope.placement(device_type, "0:0"):
        sampled_weight = flow.identity(sampled_weight_out)
        loss = flow.math.square(sampled_weight)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
        ).minimize(loss)

            #flow.watch(fc7_weight, test_global_storage.Setter("x"))
            #flow.watch_diff(fc7_weight, test_global_storage.Setter("x_diff"))
            #flow.watch_diff(
            #    sampled_weight, test_global_storage.Setter("sampled_weight_diff")
            #)
        return fc7_weight, mapped_label_out, sampled_label_out, sampled_weight

    # fake labels
    labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(
        type_name_to_np_type[label_type]
    )
    np.save("labels",labels)
    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    weight, maped_label, sampled_label, sampled_weight = PartialFcJob(labels).get()

    gpu_num = 4
    device_class_num = num_classes / gpu_num
    device_num_sample = num_sample / gpu_num
    global_sample_labels_list = []
    np_mapped_label = []
    label_map = {}
    for i in range(gpu_num):
        lower = i * device_class_num
        upper = (i + 1) * device_class_num
        condition = (labels >= lower) & (labels < upper)
        local_label = labels[condition]
        local_label = np.unique(local_label).astype(np.int32)

        idx_start = int(i * device_num_sample)
        idx_end = int((i + 1) * device_num_sample)
        local_sample_labels = sampled_label[idx_start:idx_end]
        if indexed_slice_update:
            global_sample_labels = local_sample_labels
        else:
            global_sample_labels = local_sample_labels + i * device_class_num
        global_sample_labels_list.append(global_sample_labels)

        if indexed_slice_update:
            assert (
                np.all((local_sample_labels >= lower) & (local_sample_labels < upper))
                == True
            )
        else:
            assert (
                np.all(
                    (local_sample_labels >= 0)
                    & (local_sample_labels < device_class_num)
                )
                == True
            )
        assert len(local_sample_labels) == len(np.unique(local_sample_labels))
        print("local label", local_label)
        print("local label", global_sample_labels[0 : len(local_label)])
        assert (
            np.array_equal(local_label, global_sample_labels[0 : len(local_label)])
            == True
        )
        for j in range(len(global_sample_labels)):
            label_map[global_sample_labels[j]] = j + idx_start

    for i in range(len(labels)):
        np_mapped_label.append(label_map[labels[i]])
    assert np.array_equal(np.array(np_mapped_label), maped_label.numpy()) == True

    global_sample_label = np.array(global_sample_labels_list).flatten().astype(np.int32)
    np_sample_weight = weight[global_sample_label]
    assert np.array_equal(sampled_weight.numpy(), np_sample_weight) == True

    sampled_weight_diff = test_global_storage.Get("sampled_weight_diff")
    np_weight_diff = np.zeros(weight.shape)
    for i in range(len(global_sample_label)):
        np_weight_diff[global_sample_label[i]] = sampled_weight_diff[i]

    assert np.array_equal(test_global_storage.Get("x_diff"), np_weight_diff) == True


flow.clear_default_session()


@flow.unittest.skip_unless_1n4d()
class TestPartialFc(flow.unittest.TestCase):
    def test_partial_fc(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["label_type"] = ["int32"]
        arg_dict["num_classes"] = [200]
        arg_dict["device_num_sample"] = [64]
        arg_dict["batch_size"] = [32]
        arg_dict["indexed_slice_update"] = [False]
        for arg in GenArgList(arg_dict):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
