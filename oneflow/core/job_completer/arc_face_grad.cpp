#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_arc_face_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf arc_face_grad_op;
    arc_face_grad_op.set_name("System-AutoGrad-" + op.op_name());
    ArcFaceGradOpConf* conf = arc_face_grad_op.mutable_arc_face_grad_conf();
    conf->set_depth(op.op_conf().arc_face_conf().depth());
    conf->set_margin(op.op_conf().arc_face_conf().margin());
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_label(GenLogicalBlobName(op.BnInOp2Lbi("label")));
    conf->set_sin_theta_data(GenLogicalBlobName(op.BnInOp2Lbi("sin_theta_data")));
    conf->set_dx("dx");
    op_confs->push_back(arc_face_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(arc_face_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kArcFaceConf, &GenerateBackwardOpConf);

}  // namespace oneflow
