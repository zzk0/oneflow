#ifndef ONEFLOW_CORE_KERNEL_LEAKY_RELU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LEAKY_RELU_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LeakyReluKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LeakyReluKernel);
  LeakyReluKernel() = default;
  ~LeakyReluKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct LeakyReluKernelUtil {
  static void Forward(DeviceCtx* ctx, const int32_t n, const float alpha, const T* x, T* y);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LEAKY_RELU_KERNEL_H_