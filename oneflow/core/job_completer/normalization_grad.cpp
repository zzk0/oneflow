#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_normalization_conf());
  const NormalizationOpConf& conf = op.op_conf().normalization_conf();
  LogicalBlobId* dx_lbi = DiffLbi4BnInOp("in");
  LogicalBlobId* gamma_diff_lbi = conf.has_gamma() ? DiffLbi4BnInOp("gamma") : nullptr;
  LogicalBlobId* beta_diff_lbi = conf.has_beta() ? DiffLbi4BnInOp("beta") : nullptr;
  CHECK(dx_lbi != nullptr || gamma_diff_lbi != nullptr || beta_diff_lbi != nullptr);
  if (conf.is_training()) {  
    OperatorConf normalization_grad_op;
    normalization_grad_op.set_name("System-AutoGrad-" + op.op_name());
    NormalizationGradOpConf* grad_conf = normalization_grad_op.mutable_normalization_grad_conf();
    grad_conf->set_axis(conf.axis());
    grad_conf->set_epsilon(conf.epsilon());
    grad_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    grad_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    grad_conf->set_mean(GenLogicalBlobName(op.BnInOp2Lbi("mean")));
    grad_conf->set_inv_variance(GenLogicalBlobName(op.BnInOp2Lbi("inv_variance")));
    if (conf.has_gamma()) { grad_conf->set_gamma(GenLogicalBlobName(op.BnInOp2Lbi("gamma"))); }
    if (dx_lbi != nullptr) {
      grad_conf->set_dx("dx");
      dx_lbi->set_op_name(normalization_grad_op.name());
      dx_lbi->set_blob_name(grad_conf->dx());
    }
    if (gamma_diff_lbi != nullptr) {
      grad_conf->set_gamma_diff("gamma_diff");
      gamma_diff_lbi->set_op_name(normalization_grad_op.name());
      gamma_diff_lbi->set_blob_name(grad_conf->gamma_diff());
    }
    if (beta_diff_lbi != nullptr) {
      grad_conf->set_beta_diff("beta_diff");
      beta_diff_lbi->set_op_name(normalization_grad_op.name());
      beta_diff_lbi->set_blob_name(grad_conf->beta_diff());
    }
    op_confs->emplace_back(normalization_grad_op);
  } else {
    const int32_t in_axis = LogicalBlobDesc4BnInOp("in").shape().NumAxes();
    if (dx_lbi != nullptr || gamma_diff_lbi != nullptr) {
      OperatorConf rsqrt_op;
      rsqrt_op.set_name("System-AutoGrad-" + op.op_name() + "-InvVarianceRsqrt");
      RsqrtOpConf* rsqrt_conf = rsqrt_op.mutable_rsqrt_conf();
      rsqrt_conf->set_in(GenLogicalBlobName(op.BnInOp2Lbi("moving_variance")));
      rsqrt_conf->set_out("out");
      rsqrt_conf->set_epsilon(conf.epsilon());
      op_confs->push_back(rsqrt_op);

      OperatorConf reshape_inv_var_op;
      reshape_inv_var_op.set_name("System-AutoGrad-" + op.op_name() + "-ReshapeInvVariance");
      ReshapeOpConf* reshape_inv_var_op_conf = reshape_inv_var_op.mutable_reshape_conf();
      reshape_inv_var_op_conf->set_out("out");
      reshape_inv_var_op_conf->set_in(rsqrt_op.name() + "/out");
      FOR_RANGE(size_t, i, 0, in_axis) { 
        if (i != conf.axis()){
          reshape_inv_var_op_conf->mutable_shape()->add_dim(1);
        } else{
          reshape_inv_var_op_conf->mutable_shape()->add_dim(LogicalBlobDesc4BnInOp("in").shape().At(conf.axis()));
        }
      }
      op_confs->push_back(reshape_inv_var_op);

      if (gamma_diff_lbi != nullptr) {
        OperatorConf reshape_mean_op;
        reshape_mean_op.set_name("System-AutoGrad-" + op.op_name() + "-ReshapeMean");
        ReshapeLikeOpConf* reshape_mean_op_conf = reshape_mean_op.mutable_reshape_like_conf();
        reshape_mean_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("moving_mean")));
        reshape_mean_op_conf->set_like(reshape_inv_var_op.name() + "/out");
        reshape_mean_op_conf->set_y("y");
        op_confs->push_back(reshape_mean_op);

        //normalized
        OperatorConf normalized_broadcast_sub_mean_op;
        normalized_broadcast_sub_mean_op.set_name("System-AutoGrad-" + op.op_name() + "-NormalizedBroadcastSubMean");
        BroadcastSubOpConf* normalized_broadcast_sub_mean_op_conf = normalized_broadcast_sub_mean_op.mutable_broadcast_sub_conf();
        normalized_broadcast_sub_mean_op_conf->set_a(GenLogicalBlobName(op.BnInOp2Lbi("in")));
        normalized_broadcast_sub_mean_op_conf->set_b(reshape_mean_op.name() + "/y");
        normalized_broadcast_sub_mean_op_conf->set_out("out");
        op_confs->push_back(normalized_broadcast_sub_mean_op);

        OperatorConf normalized_broadcast_mul_inv_var_op;
        normalized_broadcast_mul_inv_var_op.set_name("System-AutoGrad-" + op.op_name() + "-NormalizedBroadcastMulInvVariance");
        BroadcastMulOpConf* normalized_broadcast_mul_inv_var_op_conf = normalized_broadcast_mul_inv_var_op.mutable_broadcast_mul_conf();
        normalized_broadcast_mul_inv_var_op_conf->set_a(reshape_inv_var_op.name() + "/out");
        normalized_broadcast_mul_inv_var_op_conf->set_b(normalized_broadcast_sub_mean_op.name() + "/out");
        normalized_broadcast_mul_inv_var_op_conf->set_out("out");
        op_confs->push_back(normalized_broadcast_mul_inv_var_op);

        OperatorConf broadcast_mul_normalize_op;
        broadcast_mul_normalize_op.set_name("System-AutoGrad-" + op.op_name() + "-BroadcastMulNormalize");
        BroadcastMulOpConf* broadcast_mul_normalize_op_conf = broadcast_mul_normalize_op.mutable_broadcast_mul_conf();
        broadcast_mul_normalize_op_conf->set_a(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
        broadcast_mul_normalize_op_conf->set_b(normalized_broadcast_mul_inv_var_op.name() + "/out");
        broadcast_mul_normalize_op_conf->set_out("out");
        op_confs->push_back(broadcast_mul_normalize_op);     
        
        OperatorConf reduce_sum_gamma_diff_op;
        reduce_sum_gamma_diff_op.set_name("System-AutoGrad-" + op.op_name() + "-ReduceSum-GammaDiff");
        ReduceSumOpConf* reduce_sum_gamma_diff_op_conf = reduce_sum_gamma_diff_op.mutable_reduce_sum_conf();
        reduce_sum_gamma_diff_op_conf->set_in(broadcast_mul_normalize_op.name() + "/out");
        reduce_sum_gamma_diff_op_conf->set_out("out");
        FOR_RANGE(int32_t, i, 0, in_axis) {
          if (i != conf.axis()) { reduce_sum_gamma_diff_op_conf->add_axis(i); }
        }
        reduce_sum_gamma_diff_op_conf->set_keep_dims(false);
        op_confs->push_back(reduce_sum_gamma_diff_op);

        gamma_diff_lbi->set_op_name(reduce_sum_gamma_diff_op.name());
        gamma_diff_lbi->set_blob_name("out");
      }

      if (dx_lbi != nullptr) {
        OperatorConf reshape_gamma_op;
        reshape_gamma_op.set_name("System-AutoGrad-" + op.op_name() + "-ReshapeGamma");
        ReshapeLikeOpConf* reshape_gamma_op_conf = reshape_gamma_op.mutable_reshape_like_conf();
        reshape_gamma_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("gamma")));
        reshape_gamma_op_conf->set_like(reshape_inv_var_op.name() + "/out");
        reshape_gamma_op_conf->set_y("y");
        op_confs->push_back(reshape_gamma_op);

        OperatorConf broadcast_mul_gamma_op;
        broadcast_mul_gamma_op.set_name("System-AutoGrad-" + op.op_name() + "-BroadcastMulGamma");
        BroadcastMulOpConf* broadcast_mul_gamma_op_conf = broadcast_mul_gamma_op.mutable_broadcast_mul_conf();
        broadcast_mul_gamma_op_conf->set_a(reshape_gamma_op.name() + "/y");
        broadcast_mul_gamma_op_conf->set_b(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
        broadcast_mul_gamma_op_conf->set_out("out");
        op_confs->push_back(broadcast_mul_gamma_op);

        OperatorConf broadcast_mul_inv_var_op;
        broadcast_mul_inv_var_op.set_name("System-AutoGrad-" + op.op_name() + "-BroadcastMulInvVar");
        BroadcastMulOpConf* broadcast_mul_inv_var_op_conf = broadcast_mul_inv_var_op.mutable_broadcast_mul_conf();
        broadcast_mul_inv_var_op_conf->set_a(broadcast_mul_gamma_op.name() + "/out");
        broadcast_mul_inv_var_op_conf->set_b(reshape_inv_var_op.name() + "/out");
        broadcast_mul_inv_var_op_conf->set_out("out");
        op_confs->push_back(broadcast_mul_inv_var_op);
  
        dx_lbi->set_op_name(broadcast_mul_inv_var_op.name());
        dx_lbi->set_blob_name("out");
      }
    }

    if (beta_diff_lbi != nullptr) {
        OperatorConf reduce_sum_beta_op;
        reduce_sum_beta_op.set_name("System-AutoGrad-" + op.op_name() + "-ReduceSum-BetaDiff");
        ReduceSumOpConf* reduce_sum_beta_op_conf = reduce_sum_beta_op.mutable_reduce_sum_conf();
        reduce_sum_beta_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
        reduce_sum_beta_op_conf->set_out("out");
        FOR_RANGE(int32_t, i, 0, in_axis) {
          if (i != conf.axis()) { reduce_sum_beta_op_conf->add_axis(i); }
        }
        reduce_sum_beta_op_conf->set_keep_dims(false);
        op_confs->push_back(reduce_sum_beta_op);
        beta_diff_lbi->set_op_name(reduce_sum_beta_op.name());
        beta_diff_lbi->set_blob_name("out");

    }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kNormalizationConf, &GenerateBackwardOpConf);

}  // namespace oneflow
