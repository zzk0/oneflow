#include "oneflow/core/auto_placement/df_func.h"

namespace oneflow {

namespace df {

void DifferentialDemo() {
  Tensor vec(Shape({3, 3}), [](size_t index) { return (index + 1) * 0.095; });
  FOR_RANGE(int, i, 0, 300) {
    double lr = 0.01;
    if (i > 200) { lr = 0.001; }
    const auto& x = Clone(Update(vec, lr), 2);
    Backward(Max(TensorProduct(MatrixRowSum(x.at(0)),
                               Reciprocal(MatrixColSum(x.at(1))))));
    for (double x : vec.buffer().data()) { std::cout << x << " "; }
    std::cout << std::endl;
  }
  // Tensor var(Shape({1}), 0.5);
  // FOR_RANGE(int, i, 0, 40) {
  //   const auto& x = Clone(Update(var, 0.1), 2);
  //   Backward(Add(x.at(0), Reciprocal(x.at(1))));
  //   std::cout << var.buffer().data().at(0) << std::endl;
  // }
}

}  // namespace df

}  // namespace oneflow

int main(int argc, char** argv) {
  using namespace oneflow;
  df::DifferentialDemo();
  return 0;
}
