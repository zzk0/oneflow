#include "oneflow/core/kernel/calc_iou_matrix_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BatchCalcIoUMatrixKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchCalcIoUMatrixKernel);
  BatchCalcIoUMatrixKernel() = default;
  ~BatchCalcIoUMatrixKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* batch_boxes1_blob = BnInOp2Blob("batch_boxes1");
    const Blob* boxes2_blob = BnInOp2Blob("boxes2");
    Blob* iou_matrix_blob = BnInOp2Blob("iou_matrix");

    CHECK_EQ(batch_boxes1_blob->shape().NumAxes(), 3);
    CHECK_EQ(batch_boxes1_blob->shape().At(2), 4);
    CHECK_EQ(boxes2_blob->shape().NumAxes(), 2);
    CHECK_EQ(boxes2_blob->shape().At(1), 4);
    const int32_t num_boxes1 = batch_boxes1_blob->shape().At(0) * batch_boxes1_blob->shape().At(1);
    const int32_t num_boxes2 = boxes2_blob->shape().At(0);
    const T* boxes1_ptr = batch_boxes1_blob->dptr<T>();
    const T* boxes2_ptr = boxes2_blob->dptr<T>();
    float* iou_matrix_ptr = iou_matrix_blob->mut_dptr<float>();
    CalcIoUMatrixUtil<device_type, T>::CalcIoUMatrix(ctx.device_ctx, boxes1_ptr, num_boxes1,
                                                     boxes2_ptr, num_boxes2, iou_matrix_ptr);
  }
};

template<typename T>
struct CalcIoUMatrixUtil<DeviceType::kCPU, T> {
  static void CalcIoUMatrix(DeviceCtx* ctx, const T* boxes1_ptr, const int32_t num_boxes1,
                            const T* boxes2_ptr, const int32_t num_boxes2, float* iou_matrix_ptr) {
    UNIMPLEMENTED();
  }
};

#define REGISTER_BATCH_CALC_IOU_MATRIX_KERNEL(dev, dtype)                                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBatchCalcIouMatrixConf, dev, dtype, \
                                        BatchCalcIoUMatrixKernel<dev, dtype>)

REGISTER_BATCH_CALC_IOU_MATRIX_KERNEL(DeviceType::kGPU, float);
REGISTER_BATCH_CALC_IOU_MATRIX_KERNEL(DeviceType::kGPU, double);
REGISTER_BATCH_CALC_IOU_MATRIX_KERNEL(DeviceType::kCPU, float);
REGISTER_BATCH_CALC_IOU_MATRIX_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
