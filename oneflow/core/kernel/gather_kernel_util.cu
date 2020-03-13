#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/kernel.h"
#include <assert.h>
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

Shape GetFlatShape(const ShapeView& shape, int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<typename T, typename K, typename IDX>
__global__ void GatherForwardGpu(const IDX elem_cnt, const K* indices, const T* in,
                                 const IDX gather_dim_size, T* out, const IDX offset,
                                 NdIndexOffsetHelper<IDX, 3> in_offset_helper,
                                 NdIndexOffsetHelper<IDX, 3> out_offset_helper) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, elem_cnt) {
    IDX outer_idx, indices_idx, inner_idx;
    out_offset_helper.OffsetToNdIndex(i, &outer_idx, &indices_idx, &inner_idx);
    assert(indices[indices_idx] >= 0);
    const IDX idx = indices[indices_idx] - offset;
    if (idx >= 0 && idx < gather_dim_size) {
      const IDX in_offset = in_offset_helper.NdIndexToOffset(outer_idx, idx, inner_idx);
      out[i] = in[in_offset];
    } else {
      out[i] = 0;
    }
  }
}

bool IsSafeUseIndex32(const Shape& flat_in_shape, const int64_t num_indices) {
  const int64_t in_elem_cnt = flat_in_shape.elem_cnt();
  const int64_t out_elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

}  // namespace

template<typename T, typename K>
struct GatherKernelUtilImpl<DeviceType::kGPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, const int64_t offset) {
    const int64_t out_elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    if (IsSafeUseIndex32(flat_in_shape, num_indices)) {
      NdIndexOffsetHelper<int32_t, 3> in_offset_helper(static_cast<int32_t>(flat_in_shape.At(0)),
                                                       static_cast<int32_t>(flat_in_shape.At(1)),
                                                       static_cast<int32_t>(flat_in_shape.At(2)));
      NdIndexOffsetHelper<int32_t, 3> out_offset_helper(static_cast<int32_t>(flat_in_shape.At(0)),
                                                        static_cast<int32_t>(num_indices),
                                                        static_cast<int32_t>(flat_in_shape.At(2)));
      GatherForwardGpu<T, K, int32_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              out_elem_cnt, indices, in, flat_in_shape.At(1), out, offset, in_offset_helper,
              out_offset_helper);
    } else {
      NdIndexOffsetHelper<int64_t, 3> in_offset_helper(flat_in_shape.At(0), flat_in_shape.At(1),
                                                       flat_in_shape.At(2));
      NdIndexOffsetHelper<int64_t, 3> out_offset_helper(flat_in_shape.At(0), num_indices,
                                                        flat_in_shape.At(2));
      GatherForwardGpu<T, K, int64_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              out_elem_cnt, indices, in, flat_in_shape.At(1), out, offset, in_offset_helper,
              out_offset_helper);
    }
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000
template<typename K>
struct GatherKernelUtilImpl<DeviceType::kGPU, float16, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const float16* in,
                      const Shape& flat_in_shape, float16* out, const int64_t offset) {
    GatherKernelUtilImpl<DeviceType::kGPU, half, K>::Forward(
        ctx, indices, num_indices, reinterpret_cast<const half*>(in), flat_in_shape,
        reinterpret_cast<half*>(out), offset);
  }
};
#endif

#define INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL(in_type_pair, index_type_pair)              \
  template struct GatherKernelUtilImpl<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL, GATHER_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL

}  // namespace oneflow
