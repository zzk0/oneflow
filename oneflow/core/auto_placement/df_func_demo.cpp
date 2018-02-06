#include "oneflow/core/auto_placement/df_func.h"

namespace oneflow {

namespace {

void Demo() {
  std::shared_ptr<Value> data(new Value(std::vector<double>{1}));
  DfValue W = WeightVar(data, 0.1);
  FOR_RANGE(int, i, 0, 300) {
    Backward(Square(W));
    std::cout << data->buffer().at(0) << std::endl;
  }
}

}  // namespace

}  // namespace oneflow

int main(int argc, char** argv) {
  using namespace oneflow;
  Demo();
  return 0;
}
