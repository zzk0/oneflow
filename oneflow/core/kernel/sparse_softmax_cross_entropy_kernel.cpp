#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {
template<typename T, typename K>
static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w, const int64_t lower_bound,
                        const T* dy, const K* label, T* in_diff) {
  for (int64_t i = 0; i < n; ++i) {
    const int64_t idx = label[i] - lower_bound;
    if (idx >= 0 && idx < w) { in_diff[i * w + idx] = dy[i] * (in_diff[i * w + idx] - 1); }
  }
}
}  // namespace

template<typename T, typename K>
class SparseSoftmaxCrossEntropyCpuKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyCpuKernel);
  SparseSoftmaxCrossEntropyCpuKernel() = default;
  ~SparseSoftmaxCrossEntropyCpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* prediction_blob = BnInOp2Blob("prediction");
    const Blob* label_blob = BnInOp2Blob("label");
    Blob* tmp_blob = BnInOp2Blob("fw_softmax_num");
    Blob* buf_blob = BnInOp2Blob("fw_buf");
    Blob* prob_blob = BnInOp2Blob("prob");
    Blob* out_blob = BnInOp2Blob("out");
    const int64_t n = prediction_blob->shape().At(0);
    const int64_t w = prediction_blob->shape().Count(1);
    Memset<DeviceType::kCPU>(ctx.device_ctx, out_blob->mut_dptr(), 0,
                             out_blob->ByteSizeOfDataContentField());
    SoftmaxComputeProb<DeviceType::kCPU, T>(
        ctx.device_ctx, n, w, prediction_blob->dptr<T>(), tmp_blob->mut_dptr<T>(),
        prob_blob->mut_dptr<T>(), buf_blob->mut_dptr(), buf_blob->ByteSizeOfDataContentField());
    SparseCrossEntropyKernelUtil<DeviceType::kCPU, T, K>::ComputeEntropy(
        ctx.device_ctx, n, w, prob_blob->dptr<T>(), label_blob->dptr<K>(), out_blob->mut_dptr<T>());
  }
};

template<typename T, typename K>
class SparseSoftmaxCrossEntropyGradCpuKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyGradCpuKernel);
  SparseSoftmaxCrossEntropyGradCpuKernel() = default;
  ~SparseSoftmaxCrossEntropyGradCpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* dy_blob = BnInOp2Blob("dy");
    const Blob* label_blob = BnInOp2Blob("label");
    const Blob* prob_blob = BnInOp2Blob("prob");
    Blob* dx_blob = BnInOp2Blob("dx");
    int64_t lower_bound = 0;
    if (this->kernel_conf().has_sparse_softmax_cross_entropy_grad_conf()) {
      lower_bound = this->kernel_conf().sparse_softmax_cross_entropy_grad_conf().lower_bound();
    }
    const int64_t n = dx_blob->shape().At(0);
    const int64_t w = dx_blob->shape().Count(1);
    dx_blob->CopyDataContentFrom(ctx.device_ctx, prob_blob);
    BackwardSub(ctx.device_ctx, n, w, lower_bound, dy_blob->dptr<T>(), label_blob->dptr<K>(),
                dx_blob->mut_dptr<T>());
  }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_CPU_KERNEL(dtype, ltype)      \
  NEW_REGISTER_KERNEL(OperatorConf::kSparseSoftmaxCrossEntropyConf,                  \
                      SparseSoftmaxCrossEntropyCpuKernel<dtype, ltype>)              \
      .SetIsMatchedPred([](const KernelConf& conf) {                                 \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kCPU)    \
                && (GetDataType<dtype>::value == conf.data_type())                   \
                && (GetDataType<ltype>::value                                        \
                    == conf.sparse_softmax_cross_entropy_conf().label_type()));      \
      });                                                                            \
  NEW_REGISTER_KERNEL(OperatorConf::kSparseSoftmaxCrossEntropyMs1Conf,               \
                      SparseSoftmaxCrossEntropyCpuKernel<dtype, ltype>)              \
      .SetIsMatchedPred([](const KernelConf& conf) {                                 \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kCPU)    \
                && (GetDataType<dtype>::value == conf.data_type())                   \
                && (GetDataType<ltype>::value                                        \
                    == conf.sparse_softmax_cross_entropy_conf().label_type()));      \
      });                                                                            \
  NEW_REGISTER_KERNEL(OperatorConf::kSparseSoftmaxCrossEntropyGradConf,              \
                      SparseSoftmaxCrossEntropyGradCpuKernel<dtype, ltype>)          \
      .SetIsMatchedPred([](const KernelConf& conf) {                                 \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kCPU)    \
                && (GetDataType<dtype>::value == conf.data_type())                   \
                && (GetDataType<ltype>::value                                        \
                    == conf.sparse_softmax_cross_entropy_grad_conf().label_type())); \
      });                                                                            \
  NEW_REGISTER_KERNEL(OperatorConf::kSparseSoftmaxCrossEntropyMs1GradConf,           \
                      SparseSoftmaxCrossEntropyGradCpuKernel<dtype, ltype>)          \
      .SetIsMatchedPred([](const KernelConf& conf) {                                 \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kCPU)    \
                && (GetDataType<dtype>::value == conf.data_type())                   \
                && (GetDataType<ltype>::value                                        \
                    == conf.sparse_softmax_cross_entropy_grad_conf().label_type())); \
      });

REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_CPU_KERNEL(float, int64_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_CPU_KERNEL(double, int64_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_CPU_KERNEL(float, int32_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_CPU_KERNEL(double, int32_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_CPU_KERNEL(float, int8_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_CPU_KERNEL(double, int8_t);

}  // namespace oneflow
