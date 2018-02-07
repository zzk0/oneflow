#ifndef ONEFLOW_CORE_AUTO_PLACEMENT_BUFFER_H_
#define ONEFLOW_CORE_AUTO_PLACEMENT_BUFFER_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

namespace df {

class Buffer final {
 public:
  OF_DISALLOW_MOVE(Buffer);
  Buffer(const Buffer& buffer) = default;
  explicit Buffer(const Shape& shape, double init_val)
      : shape_(shape), data_(shape_.elem_cnt()) {
    for (double& x : data_) { x = init_val; }
  }
  explicit Buffer(const Shape& shape,
                  const std::function<double(size_t)>& Getter)
      : shape_(shape), data_(shape_.elem_cnt()) {
    FOR_RANGE(int, i, 0, data_.size()) { data_.at(i) = Getter(i); }
  }
  Buffer(const Shape& shape, const std::vector<double>& data)
      : shape_(shape), data_(data) {}
  ~Buffer() = default;

  size_t Size() const { return data_.size(); }
  const Shape& shape() const { return shape_; }

  double At(size_t index) const {
    CHECK(shape_.NumAxes() == 1);
    return data_.at(index);
  }
  double At(size_t x, size_t y) const {
    CHECK(shape_.NumAxes() == 2);
    return data_.at(x * shape_.Count(1) + y);
  }
  double At(size_t x, size_t y, size_t z) const {
    CHECK(shape_.NumAxes() == 3);
    return data_.at(x * shape_.Count(1) + y * shape_.Count(2) + z);
  }

  double& At(size_t index) { return data_.at(index); }
  double& At(size_t x, size_t y) {
    CHECK(shape_.NumAxes() == 2);
    return data_.at(x * shape_.Count(1) + y);
  }
  double& At(size_t x, size_t y, size_t z) {
    CHECK(shape_.NumAxes() == 2);
    return data_.at(x * shape_.Count(1) + y * shape_.Count(2) + z);
  }

  const std::vector<double> data() const { return data_; }
  std::vector<double>* mut_data() { return &data_; }

 private:
  Shape shape_;
  std::vector<double> data_;
};

}  // namespace df

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PLACEMENT_BUFFER_H_
