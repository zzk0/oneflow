#include "oneflow/core/kernel/arc_face_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void ArcFaceForward(DeviceCtx* ctx, const Blob* in, const Blob* label, const int64_t lower_bound,
                    const T cos_m, const T sin_m, Blob* sin_theta_data, Blob* out) {
  ArcFaceKernelUtilImpl<device_type, T, K>::Forward(
      ctx, in->shape().At(0), in->shape().At(1), in->dptr<T>(), label->dptr<K>(), lower_bound,
      cos_m, sin_m, sin_theta_data->mut_dptr<T>(), out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename K>
void ArcFaceBackward(DeviceCtx* ctx, const Blob* out_diff, const int64_t lower_bound, const T cos_m,
                     const T sin_m, const Blob* label, const Blob* sin_theta_data, Blob* in_diff) {
  ArcFaceKernelUtilImpl<device_type, T, K>::Backward(
      ctx, out_diff->shape().At(0), out_diff->shape().At(1), out_diff->dptr<T>(), label->dptr<K>(),
      lower_bound, cos_m, sin_m, sin_theta_data->dptr<T>(), in_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct ArcFaceSwitchUtil final {
#define MAKE_ARC_FACE_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
#define DEFINE_ARC_FACE_STATIC_SWITCH_FUNC(func_name)                    \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_ARC_FACE_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
  DEFINE_ARC_FACE_STATIC_SWITCH_FUNC(ArcFaceForward);
  DEFINE_ARC_FACE_STATIC_SWITCH_FUNC(ArcFaceBackward);
#undef DEFINE_ARC_FACE_STATIC_SWITCH_FUNC
#undef MAKE_ARC_FACE_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void ArcFaceKernelUtil<device_type, T>::Forward(DeviceCtx* ctx, const Blob* in, const Blob* label,
                                                const int64_t lower_bound, const T cos_m,
                                                const T sin_m, Blob* sin_theta_data, Blob* out) {
  ArcFaceSwitchUtil<device_type, T>::SwitchArcFaceForward(SwitchCase(label->data_type()), ctx, in,
                                                          label, lower_bound, cos_m, sin_m,
                                                          sin_theta_data, out);
}

template<DeviceType device_type, typename T>
void ArcFaceKernelUtil<device_type, T>::Backward(DeviceCtx* ctx, const Blob* out_diff,
                                                 const int64_t lower_bound, const T cos_m,
                                                 const T sin_m, const Blob* label,
                                                 const Blob* sin_theta_data, Blob* in_diff) {
  ArcFaceSwitchUtil<device_type, T>::SwitchArcFaceBackward(SwitchCase(label->data_type()), ctx,
                                                           out_diff, lower_bound, cos_m, sin_m,
                                                           label, sin_theta_data, in_diff);
}

template<typename T, typename K>
struct ArcFaceKernelUtilImpl<DeviceType::kCPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                      const T* in, const K* label, const int64_t lower_bound, const T cos_m,
                      const T sin_m, T* sin_theta_data, T* out);
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const T* out_diff, const K* label, const int64_t lower_bound, const T cos_m,
                       const T sin_m, const T* sin_theta_data, T* in_diff);
};

template<typename T, typename K>
void ArcFaceKernelUtilImpl<DeviceType::kCPU, T, K>::Forward(
    DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num, const T* in, const K* label,
    const int64_t lower_bound, const T cos_m, const T sin_m, T* sin_theta_data, T* out) {
  UNIMPLEMENTED();
}

template<typename T, typename K>
void ArcFaceKernelUtilImpl<DeviceType::kCPU, T, K>::Backward(
    DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num, const T* out_diff,
    const K* label, const int64_t lower_bound, const T cos_m, const T sin_m,
    const T* sin_theta_data, T* in_diff) {
  UNIMPLEMENTED();
}

#define INSTANTIATE_ARC_FACE_KERNEL_UTIL_IMPL_CPU(in_type_pair, index_type_pair)          \
  template struct ArcFaceKernelUtilImpl<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                        OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARC_FACE_KERNEL_UTIL_IMPL_CPU, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_ARC_FACE_KERNEL_UTIL_IMPL_CPU

#define INSTANTIATE_ARC_FACE_KERNEL_UTIL(device_type, in_type_pair) \
  template struct ArcFaceKernelUtil<device_type, OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARC_FACE_KERNEL_UTIL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_ARC_FACE_KERNEL_UTIL

}  // namespace oneflow
