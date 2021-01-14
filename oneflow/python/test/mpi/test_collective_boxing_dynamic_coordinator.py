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

from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name()

node_list = comm.allgather(node_name)
next_ip = socket.gethostbyname(node_list[(rank + 1) % size])
shifted_ip_list = comm.allgather(next_ip)
ip_list = shifted_ip_list[1:] + shifted_ip_list[:1]

print(ip_list)
