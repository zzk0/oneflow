#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/host_vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/vm_thread.msg.h"
#include "oneflow/core/vm/naive_vm_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

enum HostInstrOpCode { kCudaMallocHostOpcode = 0, kCudaFreeHostOpcode };

typedef void (*HostInstrFunc)(VmInstruction*);
std::vector<HostInstrFunc> host_instr_table;

#define REGISTER_HOST_INSTRUCTION(op_code, function_name) \
  COMMAND({                                               \
    host_instr_table.resize(op_code + 1);                 \
    host_instr_table.at(op_code) = &function_name;        \
  })

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaMallocHostInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableLogicalObjectId, symbol);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_END(CudaMallocHostInstruction);
// clang-format on

void VmCudaMallocHost(VmInstruction* vm_instr) {
  MirroredObject* mirrored_object = nullptr;
  char* dptr = nullptr;
  size_t size = 0;
  {
    const auto& vm_stream = vm_instr->mut_vm_instr_chain()->vm_stream();
    auto parallel_num = vm_stream.vm_thread().vm_stream_rt_desc().vm_stream_desc().parallel_num();
    FlatMsgView<CudaMallocHostInstruction> view;
    CHECK(view->Match(vm_instr->mut_vm_instr_msg()->mut_vm_instruction_proto()->mut_operand()));
    size = view->size();
    FlatMsg<MirroredObjectId> mirrored_object_id;
    mirrored_object_id->__Init__(view->symbol().value(), vm_stream.parallel_id());
    auto* mirrored_object_access =
        vm_instr->mut_mirrored_object_id2access()->FindPtr(mirrored_object_id.Get());
    CHECK_NOTNULL(mirrored_object_access);
    mirrored_object = mirrored_object_access->mut_mirrored_object();
    CHECK_EQ(mirrored_object->parallel_id(), vm_stream.parallel_id());
    CHECK_EQ(mirrored_object->logical_object().parallel_id2mirrored_object().size(), parallel_num);
    CHECK(!mirrored_object->has_object_type());
  }
  CudaCheck(cudaMallocHost(&dptr, size));
  mirrored_object->mutable_cuda_mem_buffer()->__Init__(size, dptr);
}
REGISTER_HOST_INSTRUCTION(kCudaMallocHostOpcode, VmCudaMallocHost);

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaFreeHostInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableLogicalObjectId, symbol);
FLAT_MSG_VIEW_END(CudaFreeHostInstruction);
// clang-format on

void VmCudaFreeHost(VmInstruction* vm_instr) {
  MirroredObject* mirrored_object = nullptr;
  {
    const auto& vm_stream = vm_instr->mut_vm_instr_chain()->vm_stream();
    auto parallel_num = vm_stream.vm_thread().vm_stream_rt_desc().vm_stream_desc().parallel_num();
    FlatMsgView<CudaFreeHostInstruction> view;
    CHECK(view->Match(vm_instr->mut_vm_instr_msg()->mut_vm_instruction_proto()->mut_operand()));
    FlatMsg<MirroredObjectId> mirrored_object_id;
    mirrored_object_id->__Init__(view->symbol().value(), vm_stream.parallel_id());
    auto* mirrored_object_access =
        vm_instr->mut_mirrored_object_id2access()->FindPtr(mirrored_object_id.Get());
    CHECK_NOTNULL(mirrored_object_access);
    mirrored_object = mirrored_object_access->mut_mirrored_object();
    CHECK_EQ(mirrored_object->parallel_id(), vm_stream.parallel_id());
    CHECK_EQ(mirrored_object->logical_object().parallel_id2mirrored_object().size(), parallel_num);
    CHECK(!mirrored_object->has_object_type());
  }
  CudaCheck(cudaFreeHost(mirrored_object->mut_cuda_mem_buffer()->mut_data()));
  mirrored_object->clear_cuda_mem_buffer();
}
REGISTER_HOST_INSTRUCTION(kCudaFreeHostOpcode, VmCudaFreeHost);

}  // namespace

const VmStreamTypeId HostVmStreamType::kVmStreamTypeId;

void HostVmStreamType::InitVmInstructionStatus(const VmStream& vm_stream,
                                               VmInstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveVmInstrStatusQuerier) < kVmInstructionStatusBufferBytes, "");
  NaiveVmInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void HostVmStreamType::DeleteVmInstructionStatus(const VmStream& vm_stream,
                                                 VmInstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool HostVmStreamType::QueryVmInstructionStatusDone(
    const VmStream& vm_stream, const VmInstructionStatusBuffer& status_buffer) const {
  return NaiveVmInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

ObjectMsgPtr<VmInstructionMsg> HostVmStreamType::CudaMallocHost(uint64_t symbol,
                                                                size_t size) const {
  auto vm_instr_msg = ObjectMsgPtr<VmInstructionMsg>::New();
  auto* vm_instr_proto = vm_instr_msg->mutable_vm_instruction_proto();
  vm_instr_proto->set_vm_stream_type_id(kVmStreamTypeId);
  vm_instr_proto->set_opcode(HostInstrOpCode::kCudaMallocHostOpcode);
  vm_instr_proto->mutable_vm_stream_mask()->mutable_all_vm_stream_enabled();
  {
    FlatMsgView<CudaMallocHostInstruction> view(vm_instr_proto->mutable_operand());
    view->mutable_symbol()->set_value(symbol);
    view->set_size(size);
  }
  return vm_instr_msg;
}

ObjectMsgPtr<VmInstructionMsg> HostVmStreamType::CudaFreeHost(uint64_t symbol) const {
  auto vm_instr_msg = ObjectMsgPtr<VmInstructionMsg>::New();
  auto* vm_instr_proto = vm_instr_msg->mutable_vm_instruction_proto();
  vm_instr_proto->set_vm_stream_type_id(kVmStreamTypeId);
  vm_instr_proto->set_opcode(HostInstrOpCode::kCudaFreeHostOpcode);
  vm_instr_proto->mutable_vm_stream_mask()->mutable_all_vm_stream_enabled();
  {
    FlatMsgView<CudaFreeHostInstruction> view(vm_instr_proto->mutable_operand());
    view->mutable_symbol()->set_value(symbol);
  }
  return vm_instr_msg;
}

void HostVmStreamType::Run(VmInstrChain* vm_instr_chain) const {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_instr_chain->mut_vm_instruction_list(), vm_instruction) {
    auto opcode = vm_instruction->mut_vm_instr_msg()->vm_instruction_proto().opcode();
    host_instr_table.at(opcode)(vm_instruction);
  }
  auto* status_buffer = vm_instr_chain->mut_status_buffer();
  NaiveVmInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

COMMAND(RegisterVmStreamType<HostVmStreamType>());

}  // namespace oneflow
