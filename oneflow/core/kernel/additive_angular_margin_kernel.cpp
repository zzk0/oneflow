#include "oneflow/core/kernel/additive_angular_margin_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AdditiveAngularMarginKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginKernel);
  AdditiveAngularMarginKernel() = default;
  ~AdditiveAngularMarginKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override {
    if (this->op_conf().has_additive_angular_margin_conf()) {
      return this->op_conf().additive_angular_margin_conf();
    } else if (this->op_conf().has_additive_angular_margin_ms1_conf()) {
      return this->op_conf().additive_angular_margin_ms1_conf();
    } else {
      UNIMPLEMENTED();
    }
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const float margin = GetValFromPbMessage<float>(this->GetCustomizedOpConf(), "margin");
    int64_t lower_bound = 0;
    if (this->kernel_conf().has_additive_angular_margin_conf()) {
      lower_bound = this->kernel_conf().additive_angular_margin_conf().lower_bound();
    }
    BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
    Memset<device_type>(ctx.device_ctx, BnInOp2Blob("sin_theta_data")->mut_dptr<T>(), 0,
                        BnInOp2Blob("sin_theta_data")->ByteSizeOfDataContentField());
    AdditiveAngularMarginKernelUtil<device_type, T>::Forward(
        ctx.device_ctx, lower_bound, static_cast<T>(cos(margin)), static_cast<T>(sin(margin)),
        BnInOp2Blob("in"), BnInOp2Blob("label"), BnInOp2Blob("sin_theta_data"), BnInOp2Blob("out"));
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdditiveAngularMarginConf, AdditiveAngularMarginKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdditiveAngularMarginMs1Conf, AdditiveAngularMarginKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
