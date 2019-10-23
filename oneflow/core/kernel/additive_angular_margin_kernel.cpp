#include "oneflow/core/kernel/additive_angular_margin_kernel.h"
#include "oneflow/core/kernel/additive_angular_margin_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& AdditiveAngularMarginKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().additive_angular_margin_conf();
}

template<DeviceType device_type, typename T>
void AdditiveAngularMarginKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const float margin = this->op_conf().additive_angular_margin_conf().margin();
  const int64_t lower_bound = this->kernel_conf().additive_angular_margin_conf().lower_bound();
  const T cos_m = cos(margin);
  const T sin_m = sqrt(1 - cos_m * cos_m);
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
  Memset<device_type>(ctx.device_ctx, BnInOp2Blob("sin_theta_data")->mut_dptr<T>(), 0,
                      BnInOp2Blob("sin_theta_data")->ByteSizeOfDataContentField());
  AdditiveAngularMarginKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("label"), lower_bound, cos_m, sin_m,
      BnInOp2Blob("sin_theta_data"), BnInOp2Blob("out"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdditiveAngularMarginConf, AdditiveAngularMarginKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
