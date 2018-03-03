#ifndef ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_
#define ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_

#include "oneflow/core/auto_placement/tensor.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace df {

Tensor _Update(const std::string& caller, Tensor* var, double lr);
#define Update(...) _Update(__LOC__, __VA_ARGS__)

Tensor _DiffWatch(const std::string& caller, const Tensor& input,
                  const std::function<void(const Buffer& out_diff)>& Handler);
#define DiffWatch(...) _DiffWatch(__LOC__, __VA_ARGS__)

#define __LOC__ __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

Tensor _ColIndexReduce(const std::string& caller, const Tensor& input,
                       const std::vector<std::vector<int64_t>>& reduce_indexes);
#define ColIndexReduce(...) _ColIndexReduce(__LOC__, __VA_ARGS__)

Tensor _IndexReduce(const std::string& caller, const Tensor& input,
                    const std::vector<std::vector<int64_t>>& reduce_indexes);
#define IndexReduce(...) _IndexReduce(__LOC__, __VA_ARGS__)

std::vector<Tensor> _Clone(const std::string& caller, const Tensor& input,
                           size_t n);
#define Clone(...) _Clone(__LOC__, __VA_ARGS__)

Tensor _Reshape(const std::string& caller, const Tensor& input,
                const Shape& shape);
#define Reshape(...) _Reshape(__LOC__, __VA_ARGS__)

Tensor _Minus(const std::string& caller, const Tensor& input);
#define Minus(...) _Minus(__LOC__, __VA_ARGS__)

Tensor _Abs(const std::string& caller, const Tensor& input);
#define Abs(...) _Abs(__LOC__, __VA_ARGS__)

Tensor _Exp(const std::string& caller, const Tensor& input);
#define Exp(...) _Exp(__LOC__, __VA_ARGS__)

Tensor _Tanh(const std::string& caller, const Tensor& input);
#define Tanh(...) _Tanh(__LOC__, __VA_ARGS__)

Tensor Tee(const Tensor& input, Tensor* out);

Tensor _Add(const std::string& caller, const Tensor& a, const Tensor& b);
#define ADD(...) _Add(__LOC__, __VA_ARGS__)

Tensor _Sub(const std::string& caller, const Tensor& a, const Tensor& b);
#define Sub(...) _Sub(__LOC__, __VA_ARGS__)

Tensor _ElemWiseMul(const std::string& caller, const Tensor& a,
                    const Tensor& b);
#define ElemWiseMul(...) _ElemWiseMul(__LOC__, __VA_ARGS__)

Tensor _ElemWiseDiv(const std::string& caller, const Tensor& a,
                    const Tensor& b);
#define ElemWiseDiv(...) _ElemWiseDiv(__LOC__, __VA_ARGS__)

Tensor _Mul(const std::string& caller, const Tensor& a, const Tensor& b);
#define Mul(...) _Mul(__LOC__, __VA_ARGS__)

Tensor _Reciprocal(const std::string& caller, const Tensor& input);
#define Reciprocal(...) _Reciprocal(__LOC__, __VA_ARGS__)

Tensor _Max(const std::string& caller, const Tensor& a, const Tensor& b);
#define Max(...) _Max(__LOC__, __VA_ARGS__)

Tensor _Min(const std::string& caller, const Tensor& a, const Tensor& b);
#define Min(...) _Min(__LOC__, __VA_ARGS__)

Tensor _MaxElem(const std::string& caller, const Tensor& a);
#define MaxElem(...) _MaxElem(__LOC__, __VA_ARGS__)

Tensor _Relu(const std::string& caller, const Tensor& input);
#define Relu(...) _Relu(__LOC__, __VA_ARGS__)

Tensor _MinElem(const std::string& caller, const Tensor& a);
#define MinElem(...) _MinElem(__LOC__, __VA_ARGS__)

Tensor _Sum(const std::string& caller, const Tensor& a);
#define Sum(...) _Sum(__LOC__, __VA_ARGS__)

Tensor _Avg(const std::string& caller, const Tensor& a);
#define Avg(...) _Avg(__LOC__, __VA_ARGS__)

Tensor _Variance(const std::string& caller, const Tensor& a);
#define Variance(...) _Variance(__LOC__, __VA_ARGS__)

Tensor _StandardDeviation(const std::string& caller, const Tensor& a);
#define StandardDeviation(...) _StandardDeviation(__LOC__, __VA_ARGS__)

Tensor _AvgAbsDeviation(const std::string& caller, const Tensor& a);
#define AvgAbsDeviation(...) _AvgAbsDeviation(__LOC__, __VA_ARGS__)

Tensor _DoubleVariance(const std::string& caller, const Tensor& input);
#define DoubleVariance(...) _DoubleVariance(__LOC__, __VA_ARGS__)

Tensor _DoubleAvgAbsDeviation(const std::string& caller, const Tensor& input);
#define DoubleAvgAbsDeviation(...) _DoubleAvgAbsDeviation(__LOC__, __VA_ARGS__)

Tensor _Square(const std::string& caller, const Tensor& input);
#define Square(...) _Square(__LOC__, __VA_ARGS__)

Tensor _Sqrt(const std::string& caller, const Tensor& input);
#define Sqrt(...) _Sqrt(__LOC__, __VA_ARGS__)

Tensor _MatrixRowSum(const std::string& caller, const Tensor& input);
#define MatrixRowSum(...) _MatrixRowSum(__LOC__, __VA_ARGS__)

Tensor _MatrixColSum(const std::string& caller, const Tensor& input);
#define MatrixColSum(...) _MatrixColSum(__LOC__, __VA_ARGS__)

Tensor _MatrixColMax(const std::string& caller, const Tensor& input);
#define MatrixColMax(...) _MatrixColMax(__LOC__, __VA_ARGS__)

Tensor _TensorProduct(const std::string& caller, const Tensor& a,
                      const Tensor& b);
#define TensorProduct(...) _TensorProduct(__LOC__, __VA_ARGS__)

Tensor _FixedExpectation(const std::string& caller, const Tensor& a, double e);
#define FixedExpectation(...) _FixedExpectation(__LOC__, __VA_ARGS__)

Tensor _FixedMaxVal(const std::string& caller, const Tensor& a, double e);
#define FixedMaxVal(...) _FixedMaxVal(__LOC__, __VA_ARGS__)

Tensor _Backward(const std::string& caller, const Tensor& loss);
#define BackwardRun(...) _Backward(__LOC__, __VA_ARGS__)

}  // namespace df

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_
