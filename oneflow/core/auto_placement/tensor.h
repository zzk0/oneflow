#ifndef ONEFLOW_CORE_AUTO_PLACEMENT_TENSOR_H_
#define ONEFLOW_CORE_AUTO_PLACEMENT_TENSOR_H_
#include "oneflow/core/auto_placement/buffer.h"

namespace oneflow {

namespace df {

class Tensor final {
 public:
  Tensor() = default;
  Tensor(const Tensor&) = default;
  explicit Tensor(std::shared_ptr<Buffer> buffer)
      : buffer_(buffer), diff_handler_([](const Buffer&) {}) {}
  Tensor(const Shape& shape, double init)
      : buffer_(std::shared_ptr<Buffer>(new Buffer(shape, init))),
        diff_handler_([](const Buffer&) {}) {}
  Tensor(double init)
      : buffer_(std::shared_ptr<Buffer>(new Buffer(Shape({1}), init))),
        diff_handler_([](const Buffer&) {}) {}
  Tensor(const Shape& shape, const std::function<double(size_t)>& Getter)
      : buffer_(std::shared_ptr<Buffer>(new Buffer(shape, Getter))),
        diff_handler_([](const Buffer&) {}) {}

  Tensor(const Shape& shape, const std::vector<double>& data)
      : buffer_(std::shared_ptr<Buffer>(new Buffer(shape, data))),
        diff_handler_([](const Buffer&) {}) {}
  Tensor(const std::shared_ptr<Buffer>& buffer,
         const std::function<void(const Buffer&)>& diff_handler)
      : buffer_(buffer), diff_handler_(diff_handler) {}
  Tensor(Tensor tensor, const std::function<void(const Buffer&)>& diff_handler)
      : buffer_(tensor.mut_buffer_ptr()), diff_handler_(diff_handler) {}
  ~Tensor() = default;

  inline size_t Size() const { return buffer_->Size(); }
  inline const Shape& shape() const { return buffer_->shape(); }

  inline double At(size_t index) const { return buffer_->At(index); }
  inline double At(size_t x, size_t y) const { return buffer_->At(x, y); }
  inline double At(size_t x, size_t y, size_t z) const {
    return buffer_->At(x, y, z);
  }

  inline double& At(size_t index) { return buffer_->At(index); }
  inline double& At(size_t x, size_t y) { return buffer_->At(x, y); }
  inline double& At(size_t x, size_t y, size_t z) {
    return buffer_->At(x, y, z);
  }

  const Buffer& buffer() const { return *buffer_; }
  const std::shared_ptr<Buffer>& buffer_ptr() const { return buffer_; }

  std::shared_ptr<Buffer> mut_buffer_ptr() { return buffer_; }

  void HandleDiff(const Buffer& diff) const { diff_handler_(diff); }

 private:
  std::shared_ptr<Buffer> buffer_;
  std::function<void(const Buffer&)> diff_handler_;
};

}  // namespace df

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PLACEMENT_TENSOR_H_
