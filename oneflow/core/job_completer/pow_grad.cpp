#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_pow_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf pow_op;
    pow_op.set_name(op.op_name() + "_pow_grad");
    PowOpConf* pow_op_conf = pow_op.mutable_pow_conf();
    pow_op_conf->set_in(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    pow_op_conf->set_power(op.op_conf().pow_conf().power() - 1);
    pow_op_conf->set_out("out");
    op_confs->push_back(pow_op);
    OperatorConf scalar_mul_op;
    scalar_mul_op.set_name(op.op_name() + "_scalar_mul_grad");
    ScalarMulOpConf* scalar_mul_op_conf = scalar_mul_op.mutable_scalar_mul_conf();
    scalar_mul_op_conf->set_float_operand(op.op_conf().pow_conf().power());
    scalar_mul_op_conf->set_in(pow_op.name() + "/out");
    scalar_mul_op_conf->set_out("out");
    op_confs->push_back(scalar_mul_op);
    OperatorConf multiply_op;
    multiply_op.set_name(op.op_name() + "_multiply_grad");
    MultiplyOpConf* multiply_op_conf = multiply_op.mutable_multiply_conf();
    multiply_op_conf->set_in_0(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    multiply_op_conf->set_in_1(scalar_mul_op.name() + "/out");
    multiply_op_conf->set_out("out");
    op_confs->push_back(multiply_op);
    DiffLbi4BnInOp("in")->set_op_name(multiply_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(multiply_op_conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kPowConf, &GenerateBackwardOpConf);

}  // namespace oneflow
