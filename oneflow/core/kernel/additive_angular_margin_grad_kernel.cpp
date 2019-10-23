#include "oneflow/core/kernel/additive_angular_margin_grad_kernel.h"
#include "oneflow/core/kernel/additive_angular_margin_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& AdditiveAngularMarginGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().additive_angular_margin_grad_conf();
}

template<DeviceType device_type, typename T>
void AdditiveAngularMarginGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const float margin = this->op_conf().additive_angular_margin_grad_conf().margin();
  const int64_t lower_bound = this->kernel_conf().additive_angular_margin_grad_conf().lower_bound();
  const T cos_m = cos(margin);
  const T sin_m = sqrt(1 - cos_m * cos_m);
  BnInOp2Blob("dx")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("dy"));
  AdditiveAngularMarginKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, BnInOp2Blob("dy"), lower_bound, cos_m, sin_m, BnInOp2Blob("label"),
      BnInOp2Blob("sin_theta_data"), BnInOp2Blob("dx"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdditiveAngularMarginGradConf,
                           AdditiveAngularMarginGradKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
