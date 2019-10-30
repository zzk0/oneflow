#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ExpKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExpKernel);
  ExpKernel() = default;
  ~ExpKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* in_blob = BnInOp2Blob("in");
    KernelUtil<device_type, T>::Exp(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                    BnInOp2Blob("out")->mut_dptr<T>());
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kExpConf, ExpKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
