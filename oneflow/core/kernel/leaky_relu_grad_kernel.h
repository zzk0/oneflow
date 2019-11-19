#ifndef ONEFLOW_CORE_KERNEL_LEAKY_RELU_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LEAKY_RELU_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
// #include "oneflow/core/kernel/leaky_relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LeakyReluGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LeakyReluGradKernel);
  LeakyReluGradKernel() = default;
  ~LeakyReluGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct LeakyReluGradKernelUtil {
  static void Forward(DeviceCtx* ctx, const int32_t n, const float alpha, const T* y, const T* dy,
                       T* dx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LEAKY_RELU_GRAD_KERNEL_H_