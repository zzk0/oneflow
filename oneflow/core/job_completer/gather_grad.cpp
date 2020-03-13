#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_gather_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    const BlobDesc& in_logical_blob_desc = LogicalBlobDesc4BnInOp("in");
    const GatherOpConf& gather_conf = op.op_conf().gather_conf();
    const int64_t axis = gather_conf.axis() < 0
                             ? in_logical_blob_desc.shape().NumAxes() + gather_conf.axis()
                             : gather_conf.axis();
    CHECK_GE(axis, 0);
    CHECK_LT(axis, in_logical_blob_desc.shape().NumAxes());
    const int64_t num_segments = in_logical_blob_desc.shape().At(axis);
    OperatorConf unsorted_segment_sum_op;
    unsorted_segment_sum_op.set_name("System-AutoGrad-" + op.op_name());
    UnsortedSegmentSumOpConf* conf = unsorted_segment_sum_op.mutable_unsorted_segment_sum_conf();
    conf->set_axis(axis);
    conf->set_num_segments(num_segments);
    conf->set_segment_ids(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_data(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_out("out");
    conf->set_no_duplicates_in_segment_ids(gather_conf.no_duplicates_in_indices());
    op_confs->push_back(unsorted_segment_sum_op);
    DiffLbi4BnInOp("in")->set_op_name(unsorted_segment_sum_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kGatherConf, &GenerateBackwardOpConf);

}  // namespace oneflow
