#include "oneflow/core/kernel/leaky_relu_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

namespace {

template<typename T>
__global__ void LeakyReluForwardGpu(const int n, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : x[i] * alpha; }
}

} // namespace

template<typename T>
struct LeakyReluKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t n, const float alpha, const T* x, T* y) {
    LeakyReluForwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, alpha, x,
                                                                                      y);
  }
};

#define INSTANTIATE_Leaky_Relu_KERNEL_UTIL(type_cpp, type_proto) \
  template class LeakyReluKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_Leaky_Relu_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);


}  // namespace oneflow