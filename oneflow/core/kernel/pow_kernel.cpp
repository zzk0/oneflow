#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PowKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PowKernel);
  PowKernel() = default;
  ~PowKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* in_blob = BnInOp2Blob("in");
    NewKernelUtil<device_type>::Pow(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                    this->op_conf().pow_conf().power(), in_blob->dptr<T>(),
                                    BnInOp2Blob("out")->mut_dptr<T>());
  }
};

#define REGISTER_POW_KERNEL(dev, dtype) \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kPowConf, dev, dtype, PowKernel<dev, dtype>)
REGISTER_POW_KERNEL(DeviceType::kGPU, float);
REGISTER_POW_KERNEL(DeviceType::kGPU, double);

}  // namespace oneflow
