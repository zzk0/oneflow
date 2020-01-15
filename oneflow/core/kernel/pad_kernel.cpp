#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/memory_copier.h"
namespace oneflow {

template<DeviceType device_type, typename T>
class PadKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PadKernel);
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), 0,
                        out_blob->ByteSizeOfDataContentField());
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(in_blob->data_type()));
    MemoryCopyNdDesc memory_copy_nd_desc;
    memory_copy_nd_desc.dst_shape =
        Shape({out_blob->shape().At(0) * out_blob->shape().At(1), out_blob->shape().At(2),
               out_blob->shape().At(3) * size_of_data_type});
    memory_copy_nd_desc.src_shape =
        Shape({in_blob->shape().At(0) * in_blob->shape().At(1), in_blob->shape().At(2),
               in_blob->shape().At(3) * size_of_data_type});
    const int64_t pad_top = this->op_conf().pad_conf().pad_top();
    const int64_t pad_left = this->op_conf().pad_conf().pad_left();

    std::vector<int64_t> dst_pos_vec = {0, pad_top, pad_left * size_of_data_type};
    std::vector<int64_t> src_pos_vec = {0, 0, 0};
    memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
    memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
    memory_copy_nd_desc.extent = memory_copy_nd_desc.src_shape;
    std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
    device_memory_copier->Copy(ctx.device_ctx, out_blob->mut_dptr(), in_blob->dptr(),
                               memory_copy_nd_desc);
  }
};

#define REGISTER_PAD_KERNEL(dev, dtype) \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kPadConf, dev, dtype, PadKernel<dev, dtype>)
REGISTER_PAD_KERNEL(DeviceType::kGPU, float);
REGISTER_PAD_KERNEL(DeviceType::kGPU, double);

}  // namespace oneflow
