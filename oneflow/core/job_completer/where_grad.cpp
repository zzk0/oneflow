#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_where_conf());
  if (DiffLbi4BnInOp("x") != nullptr || DiffLbi4BnInOp("y") != nullptr) {
    OperatorConf where_grad_op;
    where_grad_op.set_name(op.op_name() + "_grad");
    WhereGradOpConf* where_grad_op_conf = where_grad_op.mutable_where_grad_conf();
    where_grad_op_conf->set_out_diff(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    where_grad_op_conf->set_x_diff("y_diff");
    where_grad_op_conf->set_y_diff("x_diff");
    op_confs->push_back(where_grad_op);
    DiffLbi4BnInOp("x")->set_op_name(where_grad_op.name());
    DiffLbi4BnInOp("x")->set_blob_name("x_diff");
    DiffLbi4BnInOp("y")->set_op_name(where_grad_op.name());
    DiffLbi4BnInOp("y")->set_blob_name("y_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kWhereConf, &GenerateBackwardOpConf);

}  // namespace oneflow
