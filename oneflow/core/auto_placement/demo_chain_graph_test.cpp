#include "oneflow/core/auto_placement/demo_chain_graph.h"

namespace oneflow {

namespace df {

namespace test {

TEST(DemoChainGraph, simple_without_model) {
  DemoChainGraph graph([](DemoChainGraphBuilder* builder) {
    builder->Backward(builder->Op(
        "soft_max", {builder->Op("feature"), builder->Op("label")}));
  });
  ASSERT_EQ(graph.FwChainNodeNum(), 3);
  ASSERT_EQ(graph.ChainNodeNum(), 3 * 2);
}

TEST(DemoChainGraph, simple_with_model) {
  DemoChainGraph graph([](DemoChainGraphBuilder* builder) {
    builder->Backward(
        builder->Op("op0", {builder->Op("data"), builder->Model("model")}));
  });
  ASSERT_EQ(graph.FwChainNodeNum(), 2);
  ASSERT_EQ(graph.ChainNodeNum(), 6);
}

}  // namespace test

}  // namespace df

}  // namespace oneflow
