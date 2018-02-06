#include "oneflow/core/auto_placement/df_func.h"

namespace oneflow {

DfValue WeightVar(std::shared_ptr<Value> value, double lr) {
  return DfValue(value, [value, lr](const Value& diff) {
    CHECK(value->buffer().size() == diff.buffer().size());
    FOR_RANGE(int, i, 0, value->buffer().size()) {
      double& w = value->mut_buffer()->at(i);
      double d = diff.buffer().at(i);
      w = w - lr * d;
    }
  });
}

DfValue Square(DfValue input) {
  std::shared_ptr<Value> out(new Value(input.value().buffer()));
  for (double& x : *out->mut_buffer()) { x *= x; }
  return DfValue(out, [input](const Value& out_diff) {
    Value input_diff(input.value().buffer());
    FOR_RANGE(int, i, 0, input_diff.buffer().size()) {
      double& id = input_diff.mut_buffer()->at(i);
      double od = out_diff.buffer().at(i);
      id = 2 * id * od;
    }
    input.HandleDiff(input_diff);
  });
}

DfValue Backward(DfValue loss) {
  CHECK(loss.value().buffer().size() == 1);
  Value diff(std::vector<double>{1});
  loss.HandleDiff(diff);
  return DfValue(loss);
}

}  // namespace oneflow
