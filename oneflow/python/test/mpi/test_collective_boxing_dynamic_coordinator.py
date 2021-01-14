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
# mpirun -H oneflow-15,oneflow-16 -n 2   -mca pml ob1  --mca btl_tcp_if_include eno1  python  test_collective_boxing_dynamic_coordinator.py
from mpi4py import MPI
import socket
import oneflow as flow
import oneflow.typing as oft
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name()

node_list = comm.allgather(node_name)
next_ip = socket.gethostbyname(node_list[(rank + 1) % size])
shifted_ip_list = comm.allgather(next_ip)
ip_list = shifted_ip_list[1:] + shifted_ip_list[:1]

flow.env.machine(ip_list.join(","))
flow.env.ctrl_port(50051)

flow.clear_default_session()
flow.config.gpu_device_num(4)
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_logical_view(flow.scope.consistent_view())


@flow.global_function(function_config=func_config)
def test_job(a: oft.Numpy.Placeholder((96, 8))):
    b = flow.get_variable(
        "b",
        shape=(8, 32),
        dtype=flow.float,
        initializer=flow.random_uniform_initializer(minval=-1, maxval=1),
        trainable=True,
    )
    return flow.matmul(a, b)


print(test_job(np.random(96, 8).astype(np.float32)).get().numpy())
