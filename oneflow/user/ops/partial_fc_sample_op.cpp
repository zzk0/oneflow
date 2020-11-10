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

REGISTER_USER_OP("partial_fc_sample")
    .Input("label")
    .Output("maped_label")
    .Output("sampled_index")
    .Attr<int64_t>("num_classes")
    .Attr<int64_t>("num_sample")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const int64_t num_sample = ctx->Attr<int64_t>("num_sample");
      const user_op::TensorDesc* label = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      user_op::TensorDesc* sampled_index = ctx->TensorDesc4ArgNameAndIndex("sampled_index", 0);
      *ctx->TensorDesc4ArgNameAndIndex("maped_label", 0) = *label;
      *sampled_index = *label;
      sampled_index->mut_shape()->Set(0, num_sample);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("maped_label", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("label", 0);
      ctx->BatchAxis4ArgNameAndIndex("sampled_index", 0)->clear_value();
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* label_modifier = GetInputArgModifierFn("label", 0);
      CHECK_NOTNULL(label_modifier);
      label_modifier->set_requires_grad(false);
    });

}  // namespace oneflow
