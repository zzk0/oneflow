#include "oneflow/core/kernel/sparse_softmax_cross_entropy_grad_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_kernel.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SparseSoftmaxCrossEntropyGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* dy_blob = BnInOp2Blob("dy");
  const Blob* label_blob = BnInOp2Blob("label");
  const Blob* prob_blob = BnInOp2Blob("prob");
  Blob* dx_blob = BnInOp2Blob("dx");
  const int64_t n = dx_blob->shape().At(0);
  const int64_t w = dx_blob->shape().Count(1);
  T* dx = dx_blob->mut_dptr<T>();
  KernelUtil<device_type, T>::Copy(ctx.device_ctx, n * w, prob_blob->dptr<T>(), 1, dx, 1);
  SparseSoftmaxCrossEntropyGradKernelUtil<device_type, T, int32_t>::BackwardSub(
      ctx.device_ctx, n, w, label_blob->dptr<int32_t>(), dx);
}

template<typename T, typename K>
struct SparseSoftmaxCrossEntropyGradKernelUtil<DeviceType::kCPU, T, K> {
  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w, const K* label,
                          T* in_diff) {
    for (int64_t i = 0; i < n; ++i) { in_diff[i * w + static_cast<int64_t>(label[i])] -= 1; }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSparseSoftmaxCrossEntropyGradConf,
                           SparseSoftmaxCrossEntropyGradKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
