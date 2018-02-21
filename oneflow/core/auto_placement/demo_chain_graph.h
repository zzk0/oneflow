#ifndef ONEFLOW_CORE_AUTO_PLACEMENT_DEMO_CHAIN_GRAPH_H_
#define ONEFLOW_CORE_AUTO_PLACEMENT_DEMO_CHAIN_GRAPH_H_
#include "oneflow/core/graph/graph.h"

namespace oneflow {

namespace df {

class DemoChainNode;

class DemoChainRegst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DemoChainRegst);
  DemoChainRegst(DemoChainNode* producer, int64_t chain_regst_id)
      : producer_(producer), chain_regst_id_(chain_regst_id) {}
  ~DemoChainRegst() = default;

  const DemoChainNode* producer() const { return producer_; }
  DemoChainNode* mut_producer() { return producer_; }

  void HandleDiff(DemoChainRegst* regst) const { diff_handler_(regst); }
  void set_diff_handler(
      const std::function<void(DemoChainRegst*)>& diff_handler) {
    diff_handler_ = diff_handler;
  }

  void AddConsumer(const DemoChainNode* consumer) {
    consumers_.push_back(consumer);
  }

 private:
  DemoChainNode* producer_;
  int64_t chain_regst_id_;
  std::function<void(DemoChainRegst*)> diff_handler_;
  std::list<const DemoChainNode*> consumers_;
};

class DemoChainEdge;

class DemoChainNode final : public Node<DemoChainNode, DemoChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DemoChainNode);
  DemoChainNode(const std::string& name) : name_(name) {}
  ~DemoChainNode() = default;

  const std::string& name() const { return name_; }
  int64_t chain_node_id() const { return chain_node_id_; }

  void set_name(const std::string& name) { name_ = name; }
  void set_chain_node_id(int64_t chain_node_id) {
    chain_node_id_ = chain_node_id;
  }

 private:
  std::string name_;
  int64_t chain_node_id_;
};

class DemoChainEdge final : public Edge<DemoChainNode, DemoChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DemoChainEdge);
  DemoChainEdge() = default;
  ~DemoChainEdge() = default;
};

class DemoChainGraph : public Graph<DemoChainNode, DemoChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DemoChainGraph);
  DemoChainGraph() : chain_node_id_(-1), chain_regst_id_(-1) {}
  virtual ~DemoChainGraph() = default;
  virtual void MakeGraph();

 protected:
  DemoChainRegst* Op(const std::string& name,
                     std::vector<DemoChainRegst*> inputs);

  DemoChainRegst* Op(const std::string& name) { return Op(name, {}); }
  DemoChainRegst* Model(const std::string& name);
  void Backward(DemoChainRegst* regst);

 private:
  int64_t NewChainNodeId() { return ++chain_node_id_; }
  int64_t NewChainRegstId() { return ++chain_node_id_; }
  DemoChainRegst* NewRegst(DemoChainNode* producer);
  DemoChainNode* NewChainNode(const std::string& name);
  DemoChainNode* NewForwardNode(const std::string& name);
  DemoChainNode* NewBackwardNode(const std::string& name);
  DemoChainNode* NewDiffAccNode(const std::string& name);
  DemoChainNode* NewMdUpdtNode(const std::string& name);
  void Consume(DemoChainNode* node, DemoChainRegst* regst);

  int64_t chain_node_id_;
  int64_t chain_regst_id_;
  std::list<std::unique_ptr<DemoChainRegst>> regsts_;
};

}  // namespace df

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PLACEMENT_DEMO_CHAIN_GRAPH_H_
