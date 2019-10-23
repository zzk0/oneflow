#ifndef ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AdditiveAngularMarginKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginKernel);
  AdditiveAngularMarginKernel() = default;
  ~AdditiveAngularMarginKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_H_
