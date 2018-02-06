#ifndef ONEFLOW_CORE_AUTO_PLACEMENT_VALUE_H_
#define ONEFLOW_CORE_AUTO_PLACEMENT_VALUE_H_
#include "oneflow/core/common/util.h"

namespace oneflow {

class Value final {
 public:
  OF_DISALLOW_MOVE(Value);
  explicit Value(size_t n) : buffer_(n) {}
  explicit Value(const std::vector<double>& buffer) : buffer_(buffer) {}
  ~Value() = default;

  const std::vector<double> buffer() const { return buffer_; }
  std::vector<double>* mut_buffer() { return &buffer_; }

 private:
  std::vector<double> buffer_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PLACEMENT_VALUE_H_
