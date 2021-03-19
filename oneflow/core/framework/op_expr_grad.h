/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_EXPR_GRAD_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_EXPR_GRAD_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace one {

class OpExpr;
class OpExprGrad {
 public:
  OpExprGrad() = default;
  virtual ~OpExprGrad() = default;

  virtual Maybe<void> SaveTensorsForBackward(OpExprInterpState* ctx, const TensorTuple& inputs,
                                             const TensorTuple& outputs) = 0;

  virtual Maybe<void> DoBackward(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                                 TensorTuple* in_grads) = 0;

 protected:
  std::vector<std::shared_ptr<OpExpr>> backward_ops_;
};

class MatMulOpExprGrad : public OpExprGrad {
 public:
  MatMulOpExprGrad(const OpExpr& fw_op) {
    // Construct backward_ops_.
    // backward_ops_.push_back(...);
  }

  Maybe<void> SaveTensorsForBackward(OpExprInterpState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs) override;

  Maybe<void> DoBackward(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) override;

 private:
  std::shared_ptr<OpExpr> grad_a_op_;
  std::shared_ptr<OpExpr> grad_b_op_;
};

// 注册MatMul算子的后向
// REGISTER_USER_OP_EXPR_GRAD("matmul", MatMulOpExprGrad);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_GRAD_H_
