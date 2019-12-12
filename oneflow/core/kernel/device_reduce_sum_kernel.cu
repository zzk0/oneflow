#include "oneflow/core/kernel/reduce_sum_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuBroadcast(const int64_t elem_cnt, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { out_ptr[i] = out_ptr[0]; }
}

}  // namespace
template<typename T>
class DeviceReduceSumGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceReduceSumGpuKernel);
  DeviceReduceSumGpuKernel() = default;
  ~DeviceReduceSumGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
    const ReduceSumOpConf& conf = this->op_conf().reduce_sum_conf();
    const Shape& reduced_shape =
        conf.axis().empty()
            ? Shape::Ones(in_blob->shape().NumAxes())
            : CreateReducedShape(in_blob->shape(), {conf.axis().begin(), conf.axis().end()});
    NdarrayUtil<DeviceType::kGPU, T>::ReduceSum(
        ctx.device_ctx, XpuVarNdarray<T>(reduced_shape, out_blob->mut_dptr<T>()),
        XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
        XpuVarNdarray<T>(fw_tmp_blob, in_blob->shape().NumAxes()));
    GpuBroadcast<<<BlocksNum4ThreadsNum(out_blob->shape().elem_cnt()), kCudaThreadsNumPerBlock, 0,
                   ctx.device_ctx->cuda_stream()>>>(out_blob->shape().elem_cnt(),
                                                    out_blob->mut_dptr<T>());
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDeviceReduceSumConf, DeviceType::kGPU, float,
                                      DeviceReduceSumGpuKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDeviceReduceSumConf, DeviceType::kGPU, double,
                                      DeviceReduceSumGpuKernel<double>)

}  // namespace oneflow
