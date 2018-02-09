#include "oneflow/core/auto_placement/df_func.h"

namespace oneflow {

namespace df {

void DifferentialDemo() {
  Tensor vec(Shape({4, 4}), [](size_t index) { return index % 2 ? 0 : 1000; });
  Tensor output;
  Tensor table;
  Tensor row;
  Tensor col;
  Tensor epsilon(0.000000001);
  FOR_RANGE(int, i, 0, 2000) {
    double lr = 1;
    if (i < 400) {
      lr = 0.1;
    } else if (i < 800) {
      lr = 0.01;
    } else if (i < 1200) {
      lr = 0.001;
    } else {
      lr = 0.0001;
    }

    const auto& x =
        Clone(Tee(Add(Square(FixedExpectation(Update(&vec, lr), 1)), epsilon),
                  &output),
              3);
    const auto& load = Clone(MatrixColSum(x.at(1)), 2);
    Backward(Add(
        Max(Tee(ElemWiseMul(TensorProduct(Tee(MatrixRowSum(x.at(0)), &row),
                                          Reciprocal(Tee(load.at(0), &col))),
                            x.at(2)),
                &table)),
        Max(load.at(1))));
    std::cout << "output: ";
    for (double x : output.buffer().data()) { std::cout << x << " "; }
    std::cout << std::endl;
    std::cout << "row: ";
    for (double x : row.buffer().data()) { std::cout << x << " "; }
    std::cout << std::endl;
    std::cout << "col: ";
    for (double x : col.buffer().data()) { std::cout << x << " "; }
    std::cout << std::endl;
    std::cout << "table: ";
    for (double x : table.buffer().data()) { std::cout << x << " "; }
    std::cout << std::endl << std::endl;
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
