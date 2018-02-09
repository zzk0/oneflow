#include "oneflow/core/auto_placement/df_func.h"

namespace oneflow {

namespace df {

void DifferentialDemo() {
  Tensor var(Shape({4, 4}), [](size_t index) { return index % 2 ? 0 : 1000; });
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

    Tensor x = Add(Square(FixedExpectation(Update(&var, lr), 1)), epsilon);
    const auto& x_copies = Clone(x, 4);
    Tensor row = MatrixRowSum(x_copies.at(0));
    Tensor col = MatrixColSum(x_copies.at(1));
    Tensor table =
        ElemWiseMul(TensorProduct(row, Reciprocal(col)), x_copies.at(2));
    Tensor ii = Max(table);
    Backward(Add(ii, AvgAbsDeviation(MatrixColMax(x_copies.at(3)))));

    std::cout << "x: ";
    for (double i : x.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "row: ";
    for (double i : row.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "col: ";
    for (double i : col.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "table: ";
    for (double i : table.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl << std::endl;

    //    Backward(Variance(MatrixColMax(Update(&var, lr))));
    //    std::cout << "var: ";
    //    for (double i : var.buffer().data()) { std::cout << i << " "; }
    //    std::cout << std::endl;
  }
}

}  // namespace df

}  // namespace oneflow

int main(int argc, char** argv) {
  using namespace oneflow;
  df::DifferentialDemo();
  return 0;
}
