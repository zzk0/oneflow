#include "oneflow/core/kernel/sparse_softmax_cross_entropy_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SparseSoftmaxCrossEntropyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  Blob* tmp_blob = BnInOp2Blob("fw_softmax_num");
  Blob* buf_blob = BnInOp2Blob("fw_buf");
  Blob* prob_blob = BnInOp2Blob("prob");
  Blob* out_blob = BnInOp2Blob("out");  
  const int64_t n = prediction_blob->shape().At(0);
  const int64_t w = prediction_blob->shape().Count(1);
  SoftmaxComputeProb<device_type, T>(ctx.device_ctx, n, w, prediction_blob->dptr<T>(), tmp_blob->mut_dptr<T>(), prob_blob->mut_dptr<T>(),
                                            buf_blob->mut_dptr(),
                                            buf_blob->ByteSizeOfDataContentField());
  SparseCrossEntropyKernelUtil<device_type, T, int32_t>::ComputeEntropy(
      ctx.device_ctx, n, w, prob_blob->dptr<T>(), label_blob->dptr<int32_t>(), out_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSparseSoftmaxCrossEntropyConf,
                                         SparseSoftmaxCrossEntropyKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
