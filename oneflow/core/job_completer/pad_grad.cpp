#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_pad_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_grad");
    PadGradOpConf* conf = op_conf.mutable_pad_grad_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_out("out");
    conf->set_pad_left(op.op_conf().pad_conf().pad_left());
    conf->set_pad_right(op.op_conf().pad_conf().pad_right());
    conf->set_pad_top(op.op_conf().pad_conf().pad_top());
    conf->set_pad_bottom(op.op_conf().pad_conf().pad_bottom());
    op_confs->push_back(op_conf);
    DiffLbi4BnInOp("in")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kPadConf, &GenerateBackwardOpConf);

}  // namespace oneflow
