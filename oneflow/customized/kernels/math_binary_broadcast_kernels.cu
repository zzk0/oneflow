#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace user_op {

#ifdef WITH_CUDA

__device__ float MultiplyFunc(float x, float y) { return x * y; }

__device__ float MultiplyCalXDiff4GpuFloat(float x, float y, float dz) {
  return dz * y;
}

__device__ float MultiplyCalYDiff4GpuFloat(float x, float y, float dz) {
  return dz * x;
}

bool IsScalarTensor(const Tensor* tensor) {
  return tensor->shape().NumAxes() == 1 && tensor->shape().At(0) == 1;
}

bool InferTensorShape(const Tensor* tensor_x, const Tensor* tensor_y) {
  if (IsScalarTensor(tensor_x)) {
    return true;
  } else if (IsScalarTensor(tensor_y)) {
    return false;
  } else {
    size_t max_num_axes = std::max(tensor_x->shape().NumAxes(), tensor_y->shape().NumAxes());
    size_t min_num_axes = std::min(tensor_x->shape().NumAxes(), tensor_y->shape().NumAxes());
    //const auto& x_shape = CreateLeftExtendedShape(tensor_x->shape(), output_num_axes);
    //const auto& y_shape = CreateLeftExtendedShape(tensor_y->shape(), output_num_axes);
    for (float i = max_num_axes - 1; i >= max_num_axes - min_num_axes; --i) {
      CHECK(tensor_x->shape().At(i) == tensor_y->shape().At(i));
    }
    if (tensor_x->shape().elem_cnt() > tensor_y->shape().elem_cnt()) {
      return false;
    } else {
      return true;
    }
  }
}

#define MATH_BINARY_BROADCAST_GPU(func_name, fw_func, bw_func_cal_x_diff, bw_func_cal_y_diff, dtype) \
  __global__ void func_name##ForwardGpu(const int n, const dtype* x, const dtype* y, dtype* z,       \
                                        bool X2Y, int64_t elem_x, int64_t elem_y) {                  \
    if (X2Y) {                                                                                       \
      CUDA_1D_KERNEL_LOOP(i, n) {                                                                    \
        int64_t broadcast_index = i % elem_x;                                                        \
        z[i] = fw_func(x[broadcast_index], y[i]);                                                    \
      } \
    } else {                                                                                       \
      CUDA_1D_KERNEL_LOOP(i, n) { \
        int64_t broadcast_index = i % elem_y;                                                        \
        z[i] = fw_func(x[i], y[broadcast_index]);                                                    \
      } \
    }                                                                                                \
  }                                                                                                  \
  void func_name##Forward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y,            \
                          Tensor* tensor_z) {                                                        \
    const dtype* x = tensor_x->dptr<dtype>();                                                        \
    const dtype* y = tensor_y->dptr<dtype>();                                                        \
    dtype* z = tensor_z->mut_dptr<dtype>();                                                          \
    int64_t n = tensor_z->shape().elem_cnt();                                                        \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                           \
    bool X2Y = InferTensorShape(tensor_x, tensor_y); \
    int64_t elem_x = tensor_x->shape().elem_cnt();                                                   \
    int64_t elem_y = tensor_y->shape().elem_cnt();                                                   \
    func_name##ForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,                     \
                            ctx->cuda_stream()>>>(n, x, y, z, X2Y, elem_x, elem_y);                  \
  } \
  __global__ void func_name##XBackwardGpu(const int n, const dtype* x, const dtype* y, \
                                          const dtype* dz, dtype *dx, bool X2Y, int64_t elem_x, int64_t elem_y) { \
    if (X2Y) { \
      CUDA_1D_KERNEL_LOOP(i, n) { \
        int64_t broadcast_index = i % elem_x; \
        dx[broadcast_index] += bw_func_cal_x_diff(x[broadcast_index], y[i], dz[i]); \
      } \
    } else { \
      CUDA_1D_KERNEL_LOOP(i, n) { \
        int64_t broadcast_index = i % elem_y; \
        dx[i] = bw_func_cal_x_diff(x[i], y[broadcast_index], dz[i]); \
      } \
    } \
  } \
  void func_name##XBackward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y, \
                            const Tensor* tensor_dz, Tensor* tensor_dx) { \
    const dtype* x = tensor_x->dptr<dtype>(); \
    const dtype* y = tensor_y->dptr<dtype>(); \
    const dtype* dz = tensor_dz->dptr<dtype>(); \
    dtype* dx = tensor_dx->mut_dptr<dtype>(); \
    int64_t n = tensor_dz->shape().elem_cnt(); \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2); \
    bool X2Y = InferTensorShape(tensor_x, tensor_y); \
    int64_t elem_x = tensor_x->shape().elem_cnt(); \
    int64_t elem_y = tensor_y->shape().elem_cnt(); \
    func_name##XBackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, \
              ctx->cuda_stream()>>>(n, x, y, dz, dx, X2Y, elem_x, elem_y); \
  } \
  __global__ void func_name##YBackwardGpu(const int n, const dtype* x, const dtype* y, \
                                          const dtype* dz, dtype* dy, bool X2Y, int64_t elem_x, int64_t elem_y) { \
    if (X2Y) { \
      CUDA_1D_KERNEL_LOOP(i, n) { \
        int64_t broadcast_index = i % elem_x; \
        dy[i] = bw_func_cal_y_diff(x[broadcast_index], y[i], dz[i]); \
      } \
    } else { \
      CUDA_1D_KERNEL_LOOP(i, n) { \
        int64_t broadcast_index = i % elem_y; \
        dy[broadcast_index] += bw_func_cal_y_diff(x[i], y[broadcast_index], dz[i]); \
      } \
    } \
  } \
  void func_name##YBackward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y, \
                            const Tensor* tensor_dz, Tensor* tensor_dy) { \
    const dtype* x = tensor_x->dptr<dtype>(); \
    const dtype* y = tensor_y->dptr<dtype>(); \
    const dtype* dz = tensor_dz->dptr<dtype>(); \
    dtype *dy = tensor_dy->mut_dptr<dtype>(); \
    int64_t n = tensor_dz->shape().elem_cnt(); \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2); \
    bool X2Y = InferTensorShape(tensor_x, tensor_y); \
    int64_t elem_x = tensor_x->shape().elem_cnt(); \
    int64_t elem_y = tensor_y->shape().elem_cnt(); \
    func_name##YBackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, \
              ctx->cuda_stream()>>>(n, x, y, dz, dy, X2Y, elem_x, elem_y); \
  }

#define MATH_BINARY_BROADCAST_GPU_FLOAT_SEQ \
  OF_PP_MAKE_TUPLE_SEQ("Multiply", Multiply)

MATH_BINARY_BROADCAST_GPU(Multiply, MultiplyFunc, MultiplyCalXDiff4GpuFloat, MultiplyCalYDiff4GpuFloat, float);

class MathBinaryBroadcastFloatKernel final : public OpKernel {
  public:
    MathBinaryBroadcastFloatKernel(KernelInitContext* ctx) : OpKernel(ctx) {}
    MathBinaryBroadcastFloatKernel() = default;
    ~MathBinaryBroadcastFloatKernel() = default;

  private:
    void Compute(KernelContext* ctx) override {
      const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
      const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
      Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
      std::string binary_broadcast_math_type = ctx->GetAttr<std::string>("binary_broadcast_math_type");

#define MATH_BINARY_BROADCAST_FORWARD(binary_broadcast_math_type_str, func_name_prefix) \
    if (binary_broadcast_math_type == binary_broadcast_math_type_str) { \
      func_name_prefix##Forward(ctx->device_ctx(), tensor_x, tensor_y, tensor_z); \
    }

      OF_PP_FOR_EACH_TUPLE(MATH_BINARY_BROADCAST_FORWARD, MATH_BINARY_BROADCAST_GPU_FLOAT_SEQ);
#undef MATH_BINARY_BROADCAST_FORWARD
    }
};

REGISTER_USER_KERNEL("binary_broadcast")
    .SetCreateFn([](KernelInitContext* ctx) { return new MathBinaryBroadcastFloatKernel(ctx); })
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
        const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
        const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
        if (ctx.device_type() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat
            && y_tensor_desc->data_type() == DataType::kFloat) {
          return true;
        }
        return false;
    });

class MathBinaryBroadcastXGradGpuFloatKernel final : public OpKernel {
  public:
    MathBinaryBroadcastXGradGpuFloatKernel(KernelInitContext* ctx) : OpKernel(ctx) {}
    MathBinaryBroadcastXGradGpuFloatKernel() = default;
    ~MathBinaryBroadcastXGradGpuFloatKernel() = default;

  private:
    void Compute(KernelContext* ctx) override {
      const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
      const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
      const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
      Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
      std::string binary_broadcast_math_type = ctx->GetAttr<std::string>("binary_broadcast_math_type");

#define MATH_BINARY_BROADCAST_BACKWARD(binary_broadcast_math_type_str, func_name_prefix) \
    if (binary_broadcast_math_type == binary_broadcast_math_type_str) { \
      func_name_prefix##XBackward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dz, tensor_dx); \
    }

      OF_PP_FOR_EACH_TUPLE(MATH_BINARY_BROADCAST_BACKWARD, MATH_BINARY_BROADCAST_GPU_FLOAT_SEQ);
#undef MATH_BINARY_BROADCAST_BACKWARD
    }
};

class MathBinaryBroadcastYGradGpuFloatKernel final : public OpKernel {
  public:
    MathBinaryBroadcastYGradGpuFloatKernel(KernelInitContext* ctx) : OpKernel(ctx) {}
    MathBinaryBroadcastYGradGpuFloatKernel() = default;
    ~MathBinaryBroadcastYGradGpuFloatKernel() = default;

  private:
    void Compute(KernelContext* ctx) override {
      const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
      const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
      const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
      Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
      std::string binary_broadcast_math_type = ctx->GetAttr<std::string>("binary_broadcast_math_type");

#define MATH_BINARY_BROADCAST_BACKWARD(binary_broadcast_math_type_str, func_name_prefix) \
    if (binary_broadcast_math_type == binary_broadcast_math_type_str) { \
      func_name_prefix##YBackward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dz, tensor_dy); \
    }

      OF_PP_FOR_EACH_TUPLE(MATH_BINARY_BROADCAST_BACKWARD, MATH_BINARY_BROADCAST_GPU_FLOAT_SEQ);
#undef MATH_BINARY_FORWARD
    }
};

REGISTER_USER_KERNEL("binary_broadcast_x_grad")
  .SetCreateFn([](KernelInitContext* ctx) { return new MathBinaryBroadcastXGradGpuFloatKernel(ctx); })
  .SetIsMatchedPred([](const KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      if (ctx.device_type() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
  });

REGISTER_USER_KERNEL("binary_broadcast_y_grad")
  .SetCreateFn([](KernelInitContext* ctx) { return new MathBinaryBroadcastYGradGpuFloatKernel(ctx); })
  .SetIsMatchedPred([](const KernelRegContext& ctx) {
      const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
      if (ctx.device_type() == DeviceType::kGPU && y_tensor_desc->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
  });

#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
