/*
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
*/
#include "oneflow/core/job/mpi_manager.h"
#include <mpi.h>

#ifdef WITH_MPI

namespace oneflow {

MPIMgr::MPIMgr() {
  int provided;
  int ret = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  if (ret == MPI_SUCCESS) {
    inited_ = true;
  } else {
    inited_ = false;
    return;
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  LOG(INFO) << "MPI init success, rank = " << rank << ", size = " << size;
}

MPIMgr::~MPIMgr() { MPI_Finalize(); }

}  // namespace oneflow

#endif  // WITH_MPI
