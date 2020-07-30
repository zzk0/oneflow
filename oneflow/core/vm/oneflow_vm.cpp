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
#include "oneflow/core/vm/oneflow_vm.h"

namespace oneflow {

OneflowVM::OneflowVM(const Resource& resource, int64_t this_machine_id)
    : vm_(ObjectMsgPtr<vm::VirtualMachine>::New(vm::MakeVmDesc(resource, this_machine_id).Get())),
      scheduler_thread_(1) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_->mut_thread_ctx_list(), thread_ctx) {
    auto thread_pool = std::make_unique<ThreadPool>(1);
    thread_pool->AddWork([thread_ctx]() { thread_ctx->LoopRun(); });
    CHECK(thread_ctx2thread_pool_.emplace(thread_ctx, std::move(thread_pool)).second);
  }
}

void OneflowVM::Run(const std::shared_ptr<vm::InstructionMsgList>& instruction_msg_list) {
  mut_vm()->Receive(instruction_msg_list.get());
  scheduler_thread_.AddWork([this] {
    while (!mut_vm()->Empty()) { mut_vm()->Schedule(); }
  });
}

OneflowVM::~OneflowVM() {
  scheduler_thread_.AddWork([this] {
    CHECK(mut_vm()->Empty());
    mut_vm()->CloseAllThreads();
  });
}

}  // namespace oneflow
