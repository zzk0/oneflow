
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/yolo_kernel_util.cuh"

namespace oneflow {

namespace {

// template<typename T>
//__global__ void SelectOutIndexes(const int32_t box_num, const T* probs_ptr,
//                                 int32_t* select_inds_ptr, int32_t* valid_num_ptr,
//                                 const int32_t probs_num, const float prob_thresh,
//                                 const int32_t max_out_boxes) {
//  int32_t index_num;
//  FOR_RANGE(int32_t, i, 0, box_num) {
//    if (probs_ptr[i * probs_num + 0] > prob_thresh) { select_inds_ptr[index_num++] = i; }
//  }
//  valid_num_ptr[0] = index_num;
//  assert(valid_num_ptr[0] <= max_out_boxes);
//}

template<typename T>
__global__ void SetOutProbs(const int32_t probs_num, const T* probs_ptr,
                            const int32_t* select_inds_ptr, const int32_t* valid_num_ptr,
                            T* out_probs_ptr, const float prob_thresh) {
  const int index_num = valid_num_ptr[0];
  CUDA_1D_KERNEL_LOOP(i, index_num * probs_num) {
    const int32_t select_index = i / probs_num;
    const int32_t probs_index = i % probs_num;
    const int32_t box_index = select_inds_ptr[select_index];
    if (probs_index == 0) {
      out_probs_ptr[select_index * probs_num + probs_index] =
          probs_ptr[box_index * probs_num + probs_index];
    } else {
      T cls_prob =
          probs_ptr[box_index * probs_num + probs_index] * probs_ptr[box_index * probs_num + 0];
      out_probs_ptr[select_index * probs_num + probs_index] = cls_prob > prob_thresh ? cls_prob : 0;
    }
  }
}

template<typename T>
__global__ void SetOutBoxes(const T* bbox_ptr, const int32_t* origin_image_info_ptr,
                            const int32_t* select_inds_ptr, const int32_t* valid_num_ptr,
                            const int32_t* anchor_boxes_size_ptr, T* out_bbox_ptr,
                            const int32_t layer_height, const int32_t layer_width,
                            const int32_t layer_nbox, const int32_t image_height,
                            const int32_t image_width) {
  int32_t new_w = 0;
  int32_t new_h = 0;
  if (((float)image_width / origin_image_info_ptr[1])
      < ((float)image_height / origin_image_info_ptr[0])) {
    new_w = image_width;
    new_h = (origin_image_info_ptr[0] * image_width) / origin_image_info_ptr[1];
  } else {
    new_h = image_height;
    new_w = (origin_image_info_ptr[1] * image_height) / origin_image_info_ptr[0];
  }
  const int index_num = valid_num_ptr[0];
  CUDA_1D_KERNEL_LOOP(i, index_num) {
    const int32_t box_index = select_inds_ptr[i];
    int32_t iw = (box_index / layer_nbox) % layer_width;
    int32_t ih = (box_index / layer_nbox) / layer_width;
    int32_t ibox = box_index % layer_nbox;
    float box_x = (bbox_ptr[box_index * 4 + 0] + iw) / layer_width;
    float box_y = (bbox_ptr[box_index * 4 + 1] + ih) / layer_height;
    float box_w =
        std::exp(bbox_ptr[box_index * 4 + 2]) * anchor_boxes_size_ptr[2 * ibox] / image_width;
    float box_h =
        std::exp(bbox_ptr[box_index * 4 + 3]) * anchor_boxes_size_ptr[2 * ibox + 1] / image_height;
    out_bbox_ptr[i * 4 + 0] =
        (box_x - (image_width - new_w) / 2.0 / image_width) / ((float)new_w / image_width);
    out_bbox_ptr[i * 4 + 1] =
        (box_y - (image_height - new_h) / 2.0 / image_height) / ((float)new_h / image_height);
    out_bbox_ptr[i * 4 + 2] = box_w * (float)image_width / new_w;
    out_bbox_ptr[i * 4 + 3] = box_h * (float)image_height / new_h;
  }
}

}  // namespace

template<typename T>
class YoloDetectGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloDetectGpuKernel);
  YoloDetectGpuKernel() = default;
  ~YoloDetectGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Memset<DeviceType::kGPU>(ctx.device_ctx, BnInOp2Blob("out_probs")->mut_dptr<T>(), 0,
                             BnInOp2Blob("out_probs")->shape().elem_cnt() * sizeof(T));
    Memset<DeviceType::kGPU>(ctx.device_ctx, BnInOp2Blob("out_bbox")->mut_dptr<T>(), 0,
                             BnInOp2Blob("out_bbox")->shape().elem_cnt() * sizeof(T));
    int32_t* anchor_boxes_size_ptr = BnInOp2Blob("anchor_boxes_size_tmp")->mut_dptr<int32_t>();
    const YoloDetectOpConf& conf = op_conf().yolo_detect_conf();
    const int32_t layer_nbox = conf.anchor_boxes_size_size();
    FOR_RANGE(int32_t, i, 0, layer_nbox) {
      KernelUtil<DeviceType::kGPU, int32_t>::Set(ctx.device_ctx, conf.anchor_boxes_size(i).width(),
                                                 anchor_boxes_size_ptr + 2 * i);
      KernelUtil<DeviceType::kGPU, int32_t>::Set(ctx.device_ctx, conf.anchor_boxes_size(i).height(),
                                                 anchor_boxes_size_ptr + 2 * i + 1);
    }

    const Blob* bbox_blob = BnInOp2Blob("bbox");
    const int32_t box_num = bbox_blob->shape().At(1);
    const int32_t probs_num = conf.num_classes() + 1;

    Blob* temp_storage_blob = BnInOp2Blob("temp_storage");
    size_t temp_storage_bytes = temp_storage_blob->shape().elem_cnt();

    int32_t max_out_boxes = box_num;
    if (conf.has_max_out_boxes()) { max_out_boxes = conf.max_out_boxes(); }

    FOR_RANGE(int32_t, im_index, 0, bbox_blob->shape().At(0)) {
      const T* probs_ptr =
          BnInOp2Blob("probs")->dptr<T>() + im_index * BnInOp2Blob("probs")->shape().Count(1);
      const T* bbox_ptr =
          BnInOp2Blob("bbox")->dptr<T>() + im_index * BnInOp2Blob("bbox")->shape().Count(1);
      T* out_bbox_ptr = BnInOp2Blob("out_bbox")->mut_dptr<T>()
                        + im_index * BnInOp2Blob("out_bbox")->shape().Count(1);
      T* out_probs_ptr = BnInOp2Blob("out_probs")->mut_dptr<T>()
                         + im_index * BnInOp2Blob("out_probs")->shape().Count(1);
      // SelectOutIndexes<<<1, 1, 0, ctx.device_ctx->cuda_stream()>>>(
      //    box_num, probs_ptr,
      //    BnInOp2Blob("select_inds")->mut_dptr<int32_t>(),
      //    BnInOp2Blob("valid_num")->mut_dptr<int32_t>() + im_index *
      //    BnInOp2Blob("valid_num")->shape().Count(1), probs_num, conf.prob_thresh(),
      //    max_out_boxes);
      CudaCheck(SelectOutIndexes(ctx.device_ctx->cuda_stream(), probs_ptr,
                                 temp_storage_blob->mut_dptr<char>(),
                                 BnInOp2Blob("select_inds")->mut_dptr<int32_t>(),
                                 BnInOp2Blob("valid_num")->mut_dptr<int32_t>()
                                     + im_index * BnInOp2Blob("valid_num")->shape().Count(1),
                                 temp_storage_bytes, box_num, probs_num, conf.prob_thresh()));
      SetOutProbs<<<BlocksNum4ThreadsNum(box_num * probs_num), kCudaThreadsNumPerBlock, 0,
                    ctx.device_ctx->cuda_stream()>>>(
          probs_num, probs_ptr, BnInOp2Blob("select_inds")->dptr<int32_t>(),
          BnInOp2Blob("valid_num")->dptr<int32_t>()
              + im_index * BnInOp2Blob("valid_num")->shape().Count(1),
          out_probs_ptr, conf.prob_thresh());
      SetOutBoxes<<<BlocksNum4ThreadsNum(box_num), kCudaThreadsNumPerBlock, 0,
                    ctx.device_ctx->cuda_stream()>>>(
          bbox_ptr,
          BnInOp2Blob("origin_image_info")->dptr<int32_t>()
              + im_index * BnInOp2Blob("origin_image_info")->shape().Count(1),
          BnInOp2Blob("select_inds")->dptr<int32_t>(),
          BnInOp2Blob("valid_num")->dptr<int32_t>()
              + im_index * BnInOp2Blob("valid_num")->shape().Count(1),
          anchor_boxes_size_ptr, out_bbox_ptr, conf.layer_height(), conf.layer_width(), layer_nbox,
          conf.image_height(), conf.image_width());
    }
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kYoloDetectConf, DeviceType::kGPU, float,
                                      YoloDetectGpuKernel<float>)
// REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kYoloDetectConf, DeviceType::kGPU, double,
//                                      YoloDetectGpuKernel<double>)

}  // namespace oneflow
