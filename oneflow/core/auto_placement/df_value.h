#ifndef ONEFLOW_CORE_AUTO_PLACEMENT_DF_VALUE_H_
#define ONEFLOW_CORE_AUTO_PLACEMENT_DF_VALUE_H_
#include "oneflow/core/auto_placement/value.h"

namespace oneflow {

class DfValue final {
 public:
  DfValue(const DfValue&) = default;
  explicit DfValue(std::shared_ptr<Value> value,
                   const std::function<void(const Value&)>& diff_handler)
      : value_(value), diff_handler_(diff_handler) {}
  ~DfValue() = default;

  const Value& value() const { return *value_; }
  void HandleDiff(const Value& diff) const { diff_handler_(diff); }

 private:
  std::shared_ptr<Value> value_;
  std::function<void(const Value&)> diff_handler_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PLACEMENT_DF_VALUE_H_
