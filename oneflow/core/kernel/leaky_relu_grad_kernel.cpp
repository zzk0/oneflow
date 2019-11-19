#include "oneflow/core/kernel/leaky_relu_grad_kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LeakyReluGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("x");
  LeakyReluGradKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, in_blob->shape().elem_cnt(), this->op_conf().leaky_relu_grad_conf().alpha(),
      in_blob->dptr<T>(), BnInOp2Blob("dy")->dptr<T>(),
      BnInOp2Blob("dx")->mut_dptr<T>());
}

template<typename T>
struct LeakyReluGradKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t n, const float alpha, const T* x, const T* dy,
                       T* dx) {
    for (int64_t i = 0; i != n; ++i) { dx[i] = x[i] > 0 ? dy[i] : dy[i] * alpha; }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLeakyReluGradConf, LeakyReluGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow