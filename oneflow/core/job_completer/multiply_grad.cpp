#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_square_conf());
  if (DiffLbi4BnInOp("in_0") != nullptr) {
    OperatorConf multiply_in_0_grad_op;
    multiply_in_0_grad_op.set_name(op.op_name() + "_multiply_in_0_grad");
    MultiplyOpConf* multiply_in_0_grad_op_conf =
        multiply_in_0_grad_op.mutable_multiply_conf();
    multiply_in_0_grad_op_conf->set_in_0(GenLogicalBlobName(op.BnInOp2Lbi("in_1")));
    multiply_in_0_grad_op_conf->set_in_1(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    multiply_in_0_grad_op_conf->set_out("out");
    op_confs->push_back(multiply_in_0_grad_op);
    DiffLbi4BnInOp("in_0")->set_op_name(multiply_in_0_grad_op.name());
    DiffLbi4BnInOp("in_0")->set_blob_name("out");
  }
  if (DiffLbi4BnInOp("in_1") != nullptr) {
    OperatorConf multiply_in_1_grad_op;
    multiply_in_1_grad_op.set_name(op.op_name() + "_multiply_in_1_grad");
    MultiplyOpConf* multiply_in_1_grad_op_conf =
        multiply_in_1_grad_op.mutable_multiply_conf();
    multiply_in_1_grad_op_conf->set_in_0(GenLogicalBlobName(op.BnInOp2Lbi("in_0")));
    multiply_in_1_grad_op_conf->set_in_1(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    multiply_in_1_grad_op_conf->set_out("out");
    op_confs->push_back(multiply_in_1_grad_op);
    DiffLbi4BnInOp("in_1")->set_op_name(multiply_in_1_grad_op.name());
    DiffLbi4BnInOp("in_1")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kMultiplyConf, &GenerateBackwardOpConf);

}  // namespace oneflow
