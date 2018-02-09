#ifndef ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_
#define ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_

#include "oneflow/core/auto_placement/tensor.h"

namespace oneflow {

namespace df {

Tensor Update(Tensor* var, double lr);

std::vector<Tensor> Clone(const Tensor& input, size_t n);

Tensor Minus(const Tensor& input);
Tensor Abs(const Tensor& input);
Tensor Exp(const Tensor& input);

Tensor Tee(const Tensor& input, Tensor* out);

Tensor Add(const Tensor& a, const Tensor& b);

Tensor Sub(const Tensor& a, const Tensor& b);

Tensor ElemWiseMul(const Tensor& a, const Tensor& b);

Tensor Reciprocal(const Tensor& input);

Tensor Max(const Tensor& a, const Tensor& b);

Tensor Max(const Tensor& a);

Tensor Sum(const Tensor& a);

Tensor Avg(const Tensor& a);

Tensor Variance(const Tensor& a);

Tensor AvgAbsDeviation(const Tensor& a);

Tensor Square(const Tensor& input);

Tensor MatrixRowSum(const Tensor& input);

Tensor MatrixColSum(const Tensor& input);

Tensor MatrixColMax(const Tensor& input);

Tensor TensorProduct(const Tensor& a, const Tensor& b);

Tensor FixedExpectation(const Tensor& a, double e);

Tensor Backward(const Tensor& loss);

}  // namespace df

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_
