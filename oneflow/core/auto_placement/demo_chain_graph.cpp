#include "oneflow/core/auto_placement/demo_chain_graph.h"

namespace oneflow {
namespace df {

DemoChainRegst* DemoChainGraph::Op(const std::string& name,
                                   std::vector<DemoChainRegst*> inputs) {
  DemoChainNode* fw_node = NewForwardNode(name);
  fw_node->set_chain_node_id(NewChainNodeId());
  for (auto input : inputs) { Consume(fw_node, input); }
  DemoChainRegst* out = NewRegst(fw_node);
  out->set_diff_handler([=](DemoChainRegst* out_diff) {
    DemoChainNode* bw_node = NewBackwardNode(name);
    bw_node->set_chain_node_id(NewChainNodeId());
    Consume(bw_node, out);
    Consume(bw_node, out_diff);
    for (auto input : inputs) {
      Consume(bw_node, input);
      input->HandleDiff(NewRegst(bw_node));
    }
  });
  return out;
}

DemoChainRegst* DemoChainGraph::Model(const std::string& name) {
  DemoChainNode* md_node = NewMdUpdtNode(name);
  DemoChainRegst* model = NewRegst(md_node);
  model->set_diff_handler([=](DemoChainRegst* model_diff) {
    DemoChainNode* acc_node = NewDiffAccNode(name);
    acc_node->set_chain_node_id(NewChainNodeId());
    Consume(acc_node, model_diff);
    DemoChainRegst* diff_acc_regst = NewRegst(acc_node);
    Consume(md_node, diff_acc_regst);
    md_node->set_chain_node_id(NewChainNodeId());
  });
  return model;
}

DemoChainRegst* DemoChainGraph::NewRegst(DemoChainNode* producer) {
  auto regst = of_make_unique<DemoChainRegst>(producer, NewChainRegstId());
  regsts_.emplace_back(std::move(regst));
  return regsts_.back().get();
}

DemoChainNode* DemoChainGraph::NewChainNode(const std::string& name) {
  auto* node = new DemoChainNode(name);
  AddAllocatedNode(node);
  node->set_name(name);
  return node;
}

DemoChainNode* DemoChainGraph::NewForwardNode(const std::string& name) {
  return NewChainNode("fw_" + name);
}

DemoChainNode* DemoChainGraph::NewBackwardNode(const std::string& name) {
  return NewChainNode("bw_" + name);
}

DemoChainNode* DemoChainGraph::NewDiffAccNode(const std::string& name) {
  return NewChainNode("diff_acc_" + name);
}

DemoChainNode* DemoChainGraph::NewMdUpdtNode(const std::string& name) {
  return NewChainNode("md_updt_" + name);
}

void DemoChainGraph::Consume(DemoChainNode* node, DemoChainRegst* regst) {
  regst->AddConsumer(node);
  Connect(regst->mut_producer(), NewEdge(), node);
}

void DemoChainGraph::Backward(DemoChainRegst* regst) {
  regst->HandleDiff(NewRegst(regst->mut_producer()));
}

void DemoChainGraph::MakeGraph() {
  Backward(Op("soft_max", {Op("fc", {Op("feature")}), Op("label")}));
}

}  // namespace df
}  // namespace oneflow
