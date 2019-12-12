#include "oneflow/core/kernel/transpose_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TransposeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Transpose<device_type, T>(ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("out"),
                            this->kernel_conf().transpose_conf().perm());
}

#define REGISTER_TRANSPOSE_KERNEL(dev, dtype)                                     \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kTransposeConf, dev, dtype, \
                                        TransposeKernel<dev, dtype>)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int32_t);
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float);
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, double);
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, float);
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, double);
}  // namespace oneflow
