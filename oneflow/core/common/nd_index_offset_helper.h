#ifndef ONEFLOW_CORE_COMMON_ND_INDEX_OFFSET_HELPER_H_
#define ONEFLOW_CORE_COMMON_ND_INDEX_OFFSET_HELPER_H_

#include "oneflow/core/common/data_type.h"
#include <cassert>

namespace oneflow {

template<typename T, int N>
class NdIndexOffsetHelper {
 public:
  template<class... Ts>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(Ts&&... args) {
    T dims[N] = {args...};
    InitStrides(dims);
  }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims) { InitStrides(dims); }

  OF_DEVICE_FUNC ~NdIndexOffsetHelper() = default;

  template<class... Ts>
  OF_DEVICE_FUNC T NdIndexToOffset(Ts&&... args) const {
    constexpr int n = sizeof...(args);
    static_assert(n <= N, "");
    T index[n] = {args...};
    T offset = 0;
#pragma unroll
    for (int i = 0; i < n; ++i) { offset += index[i] * stride_[i]; }
    return offset;
  }

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index) const {
    T offset = 0;
#pragma unroll
    for (int i = 0; i < N; ++i) { offset += index[i] * stride_[i]; }
    return offset;
  }

  OF_DEVICE_FUNC T NdIndexToOffset(T* index) const {
    return NdIndexToOffset(const_cast<const T*>(index));
  }

  OF_DEVICE_FUNC T NdIndexToOffset(int n, const T* index) const {
    assert(n <= N);
    T offset = 0;
    for (int i = 0; i < n; ++i) { offset += index[i] * stride_[i]; }
    return offset;
  }

  OF_DEVICE_FUNC T NdIndexToOffset(int n, T* index) const {
    return NdIndexToOffset(n, const_cast<const T*>(index));
  }

  template<class... Ts>
  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, Ts*... args) const {
    constexpr int n = sizeof...(args);
    static_assert(n <= N, "");
    T* index[n] = {args...};
    T remaining = offset;
#pragma unroll
    for (int i = 0; i < n; ++i) {
      *index[i] = remaining / stride_[i];
      remaining = remaining % stride_[i];
    }
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index) const {
    T remaining = offset;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      index[i] = remaining / stride_[i];
      remaining = remaining % stride_[i];
    }
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, int n, T* index) const {
    T remaining = offset;
    for (int i = 0; i < n; ++i) {
      index[i] = remaining / stride_[i];
      remaining = remaining % stride_[i];
    }
  }

 private:
  OF_DEVICE_FUNC void InitStrides(const T* dims) {
    stride_[N - 1] = 1;
    for (int i = N - 2; i >= 0; --i) { stride_[i] = dims[i + 1] * stride_[i + 1]; }
  }

  T stride_[N];
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ND_INDEX_OFFSET_HELPER_H_
