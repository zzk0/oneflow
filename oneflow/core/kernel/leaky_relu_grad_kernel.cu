#include "oneflow/core/kernel/leaky_relu_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

namespace {

template<typename T>
__global__ void LeakyReluGradForwardGpu(const int n, const float alpha, const T* x, const T* dy,
                                     T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = x[i] > 0 ? dy[i] : dy[i] * alpha; }
}
}  // namespace

template<typename T>
struct LeakyReluGradKernelUtil<DeviceType::kGPU, T> {

  static void Forward(DeviceCtx* ctx, const int32_t n, const float alpha, const T* x, const T* dy,
                       T* dx) {
    LeakyReluGradForwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, alpha, x,
                                                                                      dy, dx);
  }
};

#define INSTANTIATE_Leaky_Relu_Grad_KERNEL_UTIL(type_cpp, type_proto) \
  template class LeakyReluGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_Leaky_Relu_Grad_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow