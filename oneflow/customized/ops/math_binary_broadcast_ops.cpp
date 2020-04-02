#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("binary_broadcast")
  .Input("x")
  .Input("y")
  .Output("z")
  .Attr("binary_broadcast_math_type", UserOpAttrType::kAtString)
  .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
    Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
    Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
    Shape* z_shape = ctx->Shape4ArgNameAndIndex("z", 0);
    size_t output_num_axes = std::max(x_shape->NumAxes(), y_shape->NumAxes());
    const auto& x_broadcast_shape = CreateLeftExtendedShape(ShapeView(*x_shape), output_num_axes);
    const auto& y_broadcast_shape = CreateLeftExtendedShape(ShapeView(*y_shape), output_num_axes);
    FOR_RANGE(int64_t, i, 0, x_broadcast_shape.NumAxes()) {
      CHECK_OR_RETURN(x_broadcast_shape.At(i) == 1 || y_broadcast_shape.At(i) == 1 || x_broadcast_shape.At(i) == y_broadcast_shape.At(i));
    }
    *z_shape = x_shape->elem_cnt() > y_shape->elem_cnt() ? *x_shape : *y_shape;
    return Maybe<void>::Ok();
  })
  .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
    const int32_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("z", 0).shape().NumAxes();
    SbpSignatureBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(ctx->sbp_sig_list());
    return Maybe<void>::Ok();
  });

REGISTER_USER_OP("binary_broadcast_x_grad")
  .Input("x")
  .Input("y")
  .Input("dz")
  .Output("dx")
  .Attr("binary_broadcast_math_type", UserOpAttrType::kAtString)
  .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
    Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
    Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
    Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
    Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
    CHECK((*dz_shape==*x_shape) || (*dz_shape==*y_shape));
    *dx_shape = *x_shape;
    return Maybe<void>::Ok();
  })
  .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
    const int32_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0).shape().NumAxes();
    SbpSignatureBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(ctx->sbp_sig_list());
    return Maybe<void>::Ok();
  });

REGISTER_USER_OP("binary_broadcast_y_grad")
  .Input("x")
  .Input("y")
  .Input("dz")
  .Output("dy")
  .Attr("binary_broadcast_math_type", UserOpAttrType::kAtString)
  .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
    Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
    Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
    Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
    Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
    CHECK((*dz_shape==*x_shape) || (*dz_shape==*y_shape));
    *dy_shape = *y_shape;
    return Maybe<void>::Ok();
  })
  .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
    const int32_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0).shape().NumAxes();
    SbpSignatureBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(ctx->sbp_sig_list());
    return Maybe<void>::Ok();
  });

REGISTER_USER_OP_GRAD("binary_broadcast").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                                    user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_x_grad");
    user_op::UserOpConfWrapper binary_broadcast_grad_op = 
      builder.Op("binary_broadcast_x_grad")
        .Input("x", op.input("x", 0))
        .Input("y", op.input("y", 0))
        .Input("dz", op.GetGradTensorWithOpOutput("z", 0))
        .Output("dx")
        .Attr<std::string>("binary_broadcast_math_type", op.attr<std::string>("binary_math_type"))
        .Build();
    op.BindGradTensorWithOpInput(binary_broadcast_grad_op.output("dx", 0), "x", 0);
    AddOp(binary_broadcast_grad_op);
  }
  if (op.NeedGenGradTensor4OpInput("y", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_y_grad");
    user_op::UserOpConfWrapper binary_broadcast_grad_op = 
      builder.Op("binary_broadcast_y_grad")
        .Input("x", op.input("x", 0))
        .Input("y", op.input("y", 0))
        .Input("dz", op.GetGradTensorWithOpOutput("z", 0))
        .Output("dy")
        .Attr<std::string>("binary_broadcast_math_type", op.attr<std::string>("binary_math_type"))
        .Build();
    op.BindGradTensorWithOpInput(binary_broadcast_grad_op.output("dy", 0), "y", 0);
    AddOp(binary_broadcast_grad_op);
  }
});

}  // namespace oneflow
