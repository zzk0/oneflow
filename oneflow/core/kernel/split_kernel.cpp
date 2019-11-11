#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SplitKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SplitKernel);
  SplitKernel() = default;
  ~SplitKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const int32_t axis = this->op_conf().split_conf().axis();
    const Blob* in_blob = BnInOp2Blob("in");
    const int64_t row_num = in_blob->shape().elem_cnt() / in_blob->shape().Count(axis);
    const int64_t in_col_num = in_blob->shape().Count(axis);
    int64_t in_col_offset = 0;
    for (const auto& output_bn : this->op_attribute().output_bns()) {
      Blob* out_blob = BnInOp2Blob(output_bn);
      const int64_t out_col_num = out_blob->shape().Count(axis);
      CHECK_EQ(out_blob->shape().elem_cnt(), row_num * out_col_num);
      CHECK_EQ(in_blob->data_type(), out_blob->data_type());
      KernelUtil<device_type, T>::CopyColsRegion(ctx.device_ctx, row_num, out_col_num,
                                                 in_blob->dptr<T>(), in_col_offset, in_col_num,
                                                 out_blob->mut_dptr<T>(), 0, out_col_num);
      in_col_offset += out_col_num;
    }
    CHECK_EQ(in_col_offset, in_col_num);
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSplitConf, SplitKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
