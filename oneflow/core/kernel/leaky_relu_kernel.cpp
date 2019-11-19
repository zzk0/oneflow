#include "oneflow/core/kernel/leaky_relu_kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LeakyReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  LeakyReluKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, in_blob->shape().elem_cnt(), this->op_conf().leaky_relu_conf().alpha(),
      in_blob->dptr<T>(), BnInOp2Blob("out")->mut_dptr<T>());
}

template<typename T>
struct LeakyReluKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t n, const float alpha, const T* x, T* y) {
    for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], x[i] * alpha); }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLeakyReluConf, LeakyReluKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow