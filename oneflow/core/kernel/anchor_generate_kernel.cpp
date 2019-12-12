#include "oneflow/core/kernel/kernel.h"
#include <cfenv>

namespace oneflow {

template<typename T>
class AnchorGenerateKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AnchorGenerateKernel);
  AnchorGenerateKernel() = default;
  ~AnchorGenerateKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const AnchorGenerateOpConf& conf = op_conf().anchor_generate_conf();
    const Blob* images_blob = BnInOp2Blob("images");
    const int64_t batch_height = images_blob->shape().At(1);
    const int64_t batch_width = images_blob->shape().At(2);

    Blob* anchors = BnInOp2Blob("anchors");
    Memset<DeviceType::kCPU>(ctx.device_ctx, anchors->mut_dptr<T>(), 0,
                             anchors->ByteSizeOfDataContentField());
    const float fm_stride = static_cast<float>(conf.feature_map_stride());
    const int32_t feature_map_height = std::ceil(static_cast<float>(batch_height) / fm_stride);
    const int32_t feature_map_width = std::ceil(static_cast<float>(batch_width) / fm_stride);
    auto scales_vec = PbRf2StdVec(conf.anchor_scales());
    auto ratios_vec = PbRf2StdVec(conf.aspect_ratios());
    const size_t num_anchors =
        GenerateAnchors(fm_stride, feature_map_height, feature_map_width, conf.base_anchor_height(),
                        conf.base_anchor_width(), conf.anchor_x_offset(), conf.anchor_y_offset(),
                        conf.anchor_x_stride(), conf.anchor_y_stride(), scales_vec, ratios_vec,
                        anchors->mut_dptr<T>());
    CHECK_LE(num_anchors, anchors->static_shape().At(0));
  }

  size_t GenerateAnchors(float feature_map_stride, int32_t feature_map_height,
                         int32_t feature_map_width, float base_anchor_height,
                         float base_anchor_width, float anchor_x_offset, float anchor_y_offset,
                         float anchor_x_stride, float anchor_y_stride,
                         const std::vector<float>& scales_vec, const std::vector<float>& ratios_vec,
                         T* anchors_ptr) const {
    const float base_x_ctr = anchor_x_offset;
    const float base_y_ctr = anchor_y_offset;
    const size_t num_anchors = scales_vec.size();
    std::vector<T> base_anchors_vec(num_anchors * 4);

    FOR_RANGE(int32_t, i, 0, ratios_vec.size()) {
      const float ratio_sqrt = std::sqrt(ratios_vec.at(i));
      const float height = scales_vec.at(i) / ratio_sqrt * base_anchor_height;
      const float width = scales_vec.at(i) * ratio_sqrt * base_anchor_width;

      const int32_t cur_anchor_idx = i;
      base_anchors_vec[cur_anchor_idx * 4 + 0] = base_y_ctr - 0.5 * height;  // y1
      base_anchors_vec[cur_anchor_idx * 4 + 1] = base_x_ctr - 0.5 * width;   // x1
      base_anchors_vec[cur_anchor_idx * 4 + 2] = base_y_ctr + 0.5 * height;  // y2
      base_anchors_vec[cur_anchor_idx * 4 + 3] = base_x_ctr + 0.5 * width;   // x2
    }

    FOR_RANGE(int32_t, h, 0, feature_map_height) {
      FOR_RANGE(int32_t, w, 0, feature_map_width) {
        auto* cur_anchor_ptr = anchors_ptr + (h * feature_map_width + w) * num_anchors * 4;
        FOR_RANGE(int32_t, i, 0, num_anchors) {
          cur_anchor_ptr[i * 4 + 0] = base_anchors_vec[i * 4 + 0] + h * anchor_y_stride;  // y1
          cur_anchor_ptr[i * 4 + 1] = base_anchors_vec[i * 4 + 1] + w * anchor_x_stride;  // x1
          cur_anchor_ptr[i * 4 + 2] = base_anchors_vec[i * 4 + 2] + h * anchor_y_stride;  // y2
          cur_anchor_ptr[i * 4 + 3] = base_anchors_vec[i * 4 + 3] + w * anchor_x_stride;  // x2
        }
      }
    }
    return num_anchors * feature_map_height * feature_map_width;
  }
};

#define REGISTER_ANCHOR_GENERATE_KERNEL(dtype)                                               \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kAnchorGenerateConf, DeviceType::kCPU, \
                                        dtype, AnchorGenerateKernel<dtype>)

REGISTER_ANCHOR_GENERATE_KERNEL(float);
REGISTER_ANCHOR_GENERATE_KERNEL(double);

}  // namespace oneflow
