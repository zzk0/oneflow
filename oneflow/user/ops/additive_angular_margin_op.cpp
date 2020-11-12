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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("additive_angular_margin")
    .Input("x")
    .Input("label")
    .Output("y")
    .Output("sin_theta_data")
    .Attr<float>("margin")
    .Attr<int64_t>("depth")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* label = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      CHECK_EQ_OR_RETURN(label->shape().At(0), x->shape().At(0));
      user_op::TensorDesc* sin_theta_data = ctx->TensorDesc4ArgNameAndIndex("sin_theta_data", 0);
      *ctx->TensorDesc4ArgNameAndIndex("y", 0) = *x;
      *sin_theta_data->mut_data_type() = x->data_type();
      *sin_theta_data->mut_shape() = label->shape();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      *ctx->BatchAxis4ArgNameAndIndex("sin_theta_data", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* label_arg_modifier = GetInputArgModifierFn("label", 0);
      label_arg_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("label", 0), 0)
          .Split(user_op::OpArg("sin_theta_data", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .Build();
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 1)
          .Broadcast(user_op::OpArg("label", 0))
          .PartialSum(user_op::OpArg("sin_theta_data", 0))
          .Split(user_op::OpArg("y", 0), 1)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("additive_angular_margin_grad")
    .Input("dy")
    .Input("label")
    .Input("sin_theta_data")
    .Output("dx")
    .Attr<float>("margin")
    .Attr<int64_t>("depth")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      const user_op::TensorDesc* label = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      CHECK_EQ_OR_RETURN(label->shape().At(0), dy->shape().At(0));
      *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *dy;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("dy", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("label", 0), 0)
          .Split(user_op::OpArg("sin_theta_data", 0), 0)
          .Split(user_op::OpArg("dx", 0), 0)
          .Build();
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), 1)
          .Broadcast(user_op::OpArg("label", 0))
          .Broadcast(user_op::OpArg("sin_theta_data", 0))
          .Split(user_op::OpArg("dx", 0), 1)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("additive_angular_margin")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("additive_angular_margin_grad")
                .Input("label", op.input("label", 0))
                .Input("sin_theta_data", op.output("sin_theta_data", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr("margin", op.attr<float>("margin"))
                .Attr("depth", op.attr<int64_t>("depth"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
