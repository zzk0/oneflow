#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_clip_by_value_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_grad");
    ClipByValueGradOpConf* conf = op_conf.mutable_clip_by_value_grad_conf();
    conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_min_val(op.op_conf().clip_by_value_conf().min_val());
    conf->set_max_val(op.op_conf().clip_by_value_conf().max_val());
    conf->set_dx("dx");
    op_confs->push_back(op_conf);
    DiffLbi4BnInOp("in")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kClipByValueConf, &GenerateBackwardOpConf);

}  // namespace oneflow
