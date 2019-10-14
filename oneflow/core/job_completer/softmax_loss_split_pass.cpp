#include "oneflow/core/job_completer/softmax_loss_split_pass.h"

namespace oneflow {

namespace {

void AddSoftmaxSplitOpConf() {}

void UpdateConsumerOpConf(const std::string& new_out_lbn, const std::string& new_prob_lbn,
                          const OpNode* op_node, JobBuilder* job_builder) {
  // prob、out
  const LogicalBlobId& old_out_lbi = op_node->op().BnInOp2Lbi("out");
  const LogicalBlobId& old_prob_lbi = op_node->op().BnInOp2Lbi("prob");
  const std::string& old_out_lbn = GenLogicalBlobName(old_out_lbi);
  const std::string& old_prob_lbn = GenLogicalBlobName(old_prob_lbi);
  for (const OpEdge* edge : op_node->out_edges()) {
    OpNode* out_node = edge->dst_node();
    OperatorConf mut_op_conf(out_node->op().op_conf());
    PbMessage* mut_conf = MutableMessageInPbMessage(&mut_op_conf, mut_op_conf.op_type_case());
    if (edge->lbi2ibns().find(old_out_lbi) != edge->lbi2ibns().end()) {
      const auto& out_ibns = edge->lbi2ibns().at(old_out_lbi);
      for (const std::string& ibn : out_ibns) {
        ReplaceStrValInPbFdOrPbRpf(mut_conf, ibn, old_out_lbn, new_out_lbn);
      }
    }
    if (edge->lbi2ibns().find(old_prob_lbi) != edge->lbi2ibns().end()) {
      const auto& prob_ibns = edge->lbi2ibns().at(old_prob_lbi);
      for (const std::string& ibn : prob_ibns) {
        ReplaceStrValInPbFdOrPbRpf(mut_conf, ibn, old_prob_lbn, new_prob_lbn);
      }
    }
    job_builder->MutOpsOnlyOnce({mut_op_conf});
  }
}

void UpdateJobHelperConfProducedLbi2ConsumedDiffLbi(LogicalBlobId old_lbi, LogicalBlobId new_lbi,
                                                    JobBuilder* job_builder) {
  auto& mut_pairs =
      (*job_builder->mutable_helper()->mutable_tag2lbi_relations())[kProducedLbi2ConsumedDiffLbi];
  for (auto& mut_pair : *mut_pairs.mutable_pair()) {
    if (mut_pair.first() == old_lbi) { *mut_pair.mutable_first() = new_lbi; }
  }
}

}  // namespace

void SoftmaxLossSplitPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  HashMap<std::string, LogicalBlobId> op_name2lbi;
  op_graph.ForEachNode([&](const OpNode* node) {
    if (node->op().op_conf().has_sparse_softmax_cross_entropy_conf()) {
      if (1
          || node->SbpParallel4BnInOp("prediction").has_split_parallel()
                 && node->SbpParallel4BnInOp("prediction").split_parallel().axis() == 1) {
        const auto& sparse_softmax_cross_entropy_conf =
            node->op().op_conf().sparse_softmax_cross_entropy_conf();

        // OperatorConf reduce_max_stage0_op_conf(node->op().op_conf());
        // auto* reduce_max_stage0_conf = reduce_max_stage0_op_conf.mutable_reduce_max_conf();
        OperatorConf reduce_max_stage0_op_conf;
        reduce_max_stage0_op_conf.set_name(node->op().op_name() + "-softmax_reduce_max_stage0");
        auto* reduce_max_stage0_conf = reduce_max_stage0_op_conf.mutable_reduce_sum_conf();
        reduce_max_stage0_conf->set_in(sparse_softmax_cross_entropy_conf.prediction());
        reduce_max_stage0_conf->set_out("out");
        job_builder->AddOps(node->parallel_desc().parallel_conf(), {reduce_max_stage0_op_conf});
        // job_builder->MutOpsOnlyOnce({reduce_max_stage0_op_conf});

        OperatorConf reduce_max_stage1_op_conf;
        reduce_max_stage1_op_conf.set_name(node->op().op_name() + "-softmax_reduce_max_stage1");
        // auto* reduce_max_stage1_conf = reduce_max_stage1_op_conf.mutable_reduce_max_conf();
        auto* reduce_max_stage1_conf = reduce_max_stage1_op_conf.mutable_reduce_sum_conf();
        reduce_max_stage1_conf->set_in(reduce_max_stage0_op_conf.name() + "/out");
        reduce_max_stage1_conf->set_out("out");
        job_builder->AddOps(node->parallel_desc().parallel_conf(), {reduce_max_stage1_op_conf});

        OperatorConf broadcast_sub_op_conf;
        broadcast_sub_op_conf.set_name(node->op().op_name() + "-softmax_submax");
        auto* broadcast_sub_conf = broadcast_sub_op_conf.mutable_broadcast_sub_conf();
        broadcast_sub_conf->set_a(sparse_softmax_cross_entropy_conf.prediction());
        broadcast_sub_conf->set_b(reduce_max_stage1_op_conf.name() + "/out");
        broadcast_sub_conf->set_out("out");
        job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_sub_op_conf});

        OperatorConf exp_op_conf;
        exp_op_conf.set_name(node->op().op_name() + "-softmax_exp");
        // auto* exp_conf = exp_op_conf.mutable_exp_conf();
        auto* exp_conf = exp_op_conf.mutable_relu_conf();
        exp_conf->set_in(broadcast_sub_op_conf.name() + "/out");
        exp_conf->set_out("out");
        job_builder->AddOps(node->parallel_desc().parallel_conf(), {exp_op_conf});

        OperatorConf reduce_sum_op_conf;
        reduce_sum_op_conf.set_name(node->op().op_name() + "-softmax_reduce_sum");
        auto* reduce_sum_conf = reduce_sum_op_conf.mutable_reduce_sum_conf();
        reduce_sum_conf->set_in(exp_op_conf.name() + "/out");
        reduce_sum_conf->set_out("out");
        job_builder->AddOps(node->parallel_desc().parallel_conf(), {reduce_sum_op_conf});

        OperatorConf broadcast_div_op_conf;
        broadcast_div_op_conf.set_name(node->op().op_name() + "-softmax_div");
        auto* broadcast_div_conf = broadcast_div_op_conf.mutable_broadcast_sub_conf();
        broadcast_div_conf->set_a(exp_op_conf.name() + "/out");
        broadcast_div_conf->set_b(reduce_sum_op_conf.name() + "/out");
        broadcast_div_conf->set_out("out");
        job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_div_op_conf});

        OperatorConf sparse_cross_entropy_op_conf;
        sparse_cross_entropy_op_conf.set_name(node->op().op_name() + "-sparse_cross_entropy");
        auto* sparse_cross_entropy_conf =
            sparse_cross_entropy_op_conf.mutable_sparse_cross_entropy_conf();
        sparse_cross_entropy_conf->set_prediction(broadcast_div_op_conf.name() + "/out");
        sparse_cross_entropy_conf->set_label(sparse_softmax_cross_entropy_conf.label());
        // sparse_cross_entropy_conf->set_depth(sparse_softmax_cross_entropy_conf.depth());
        sparse_cross_entropy_conf->set_out("out");
        job_builder->AddOps(node->parallel_desc().parallel_conf(), {sparse_cross_entropy_op_conf});
        std::string out_lbn = sparse_cross_entropy_op_conf.name() + "/out";
        std::string prob_lbn = broadcast_div_op_conf.name() + "/out";
        LogicalBlobId out_lbi;
        out_lbi.set_op_name(sparse_cross_entropy_op_conf.name());
        out_lbi.set_blob_name("out");
        UpdateJobHelperConfProducedLbi2ConsumedDiffLbi(node->op().BnInOp2Lbi("out"), out_lbi,
                                                       job_builder);
        UpdateConsumerOpConf(out_lbn, prob_lbn, node, job_builder);
        job_builder->DelOps({node->op().op_conf()});
      }
    }
  });
}

}  // namespace oneflow
