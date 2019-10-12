#include "oneflow/core/operator/deconv_op.h"
#include "oneflow/core/kernel/deconv_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

template<typename T>
class DeconvGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvGPUKernel);
  DeconvGPUKernel() = default;
  ~DeconvGPUKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().deconv_conf(); }

  void VirtualKernelInit() override {
    const DeconvOpConf& op_conf = this->op_conf().deconv_conf();
    const ConvConf& conv_conf = this->op_conf().deconv_conf().conv_conf();
    const int32_t num_spatial_dims = this->op_conf().deconv_conf().conv_conf().num_spatial_dims();
    Shape x_shape(this->kernel_conf().deconv_conf().in());
    Shape y_shape(this->kernel_conf().deconv_conf().out());
    Shape weight_shape(this->kernel_conf().deconv_conf().weight());

    const std::string& data_format = conv_conf.data_format();
    this->x_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, x_shape, data_format));
    this->y_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, y_shape, data_format));
    this->filter_desc_.reset(new CudnnFilterDesc(GetDataType<T>::value, weight_shape, data_format));
    this->deconv_desc_.reset(new CudnnDeconvDesc(GetDataType<T>::value, x_shape,
                                                 this->op_conf().deconv_conf()));
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* x_blob = BnInOp2Blob("x");
    const Blob* filter_blob = BnInOp2Blob("filter");
    // const Blob* bias_blob = BnInOp2Blob("bias");
    Blob* y_blob = BnInOp2Blob("y");
    Blob* cudnn_buf = BnInOp2Blob("cudnn_buf");
    void* buf_ptr = cudnn_buf ? cudnn_buf->mut_dptr() : nullptr;
    size_t buf_size = cudnn_buf ? cudnn_buf->ByteSizeOfBlobBody() : 0;
    CudaCheck(cudnnConvolutionBackwardData(
        ctx.device_ctx->cudnn_handle(), CudnnSPOnePtr<T>(), this->filter_desc_->Get(),
        filter_blob->dptr<T>(), this->x_desc_->Get(), x_blob->dptr<T>(), this->deconv_desc_->Get(),
        static_cast<cudnnConvolutionBwdDataAlgo_t>(
            this->kernel_conf().deconv_conf().cudnn_bwd_data_algo()),
        buf_ptr, buf_size, CudnnSPZeroPtr<T>(), this->y_desc_->Get(), y_blob->mut_dptr<T>()));
  }

  mutable std::unique_ptr<CudnnTensorDesc> x_desc_;
  mutable std::unique_ptr<CudnnTensorDesc> y_desc_;
  mutable std::unique_ptr<CudnnFilterDesc> filter_desc_;
  mutable std::unique_ptr<CudnnDeconvDesc> deconv_desc_;
};

#define REGISTER_DECONV_GPU_KERNEL(dtype)                                                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDeconvConf, DeviceType::kGPU, dtype, \
                                        DeconvGPUKernel<dtype>)

REGISTER_DECONV_GPU_KERNEL(float);
REGISTER_DECONV_GPU_KERNEL(double);
REGISTER_DECONV_GPU_KERNEL(float16);

}  //  namespace oneflow
