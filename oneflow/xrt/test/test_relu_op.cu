#include "oneflow/xrt/parameter.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace oneflow {
namespace xrt {

__global__ void ReluKernel(const float *in, float *out, const size_t len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tid; i < len; i += blockDim.x * gridDim.x) {
    out[i] = (in[i] > 0.f) ? in[i] : 0.f;
  }
}

void ComputeRelu(/*void *stream, */const Parameter &input,
                 const Parameter &output) {
  int64_t length = input.shape().elem_cnt();
  CHECK_EQ(length, output.shape().elem_cnt());

  const float *in = input.data<float>();
  float *out = output.data<float>();

  int num_threads = 512;
  int num_blocks = (length + num_threads - 1) / num_threads;
  ReluKernel<<<num_blocks, num_threads>>>(in, out, length);
  CHECK_EQ(cudaSuccess, cudaPeekAtLastError());
}

}  // namespace xrt
}  // namespace oneflow
