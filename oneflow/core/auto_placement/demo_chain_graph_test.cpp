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
  std::vector<std::vector<int64_t>> expected_fw_ids{{0}, {1}, {2},
                                                    {2}, {1}, {0}};
  ASSERT_TRUE(graph.CalcChainNodeId2FwChainNodeId() == expected_fw_ids);
  std::vector<std::vector<int64_t>> expected_producer_ids{{0}, {1}, {2},
                                                          {2}, {3}, {3}};
  ASSERT_TRUE(graph.CalcChainRegstId2ProducerChainNodeId()
              == expected_producer_ids);

  std::vector<std::vector<int64_t>> expected_path{
      {0, 2, 3, 5}, {1, 2, 3, 4}, {2, 3}, {2, 3}, {3, 4}, {3, 5}};
  ASSERT_TRUE(graph.CalcChainRegstId2PathChainNodeIds() == expected_path);

  std::vector<std::vector<int64_t>> expected_splited_regst_ids{{0}, {1}, {2},
                                                               {3}, {4}, {5}};
  ASSERT_TRUE(graph.SplitedRegstIds() == expected_splited_regst_ids);

  std::vector<std::vector<int64_t>> expected_cloned_regst_ids{};
  ASSERT_TRUE(graph.ClonedRegstIds() == expected_cloned_regst_ids);
}

TEST(DemoChainGraph, simple_with_model) {
  DemoChainGraph graph([](DemoChainGraphBuilder* builder) {
    builder->Backward(
        builder->Op("op0", {builder->Op("data"), builder->Model("model")}));
  });
  ASSERT_EQ(graph.FwChainNodeNum(), 2);
  ASSERT_EQ(graph.ChainNodeNum(), 6);

  std::vector<std::vector<int64_t>> expected_fw_ids{{0}, {1}, {1},
                                                    {1}, {1}, {0}};
  ASSERT_TRUE(graph.CalcChainNodeId2FwChainNodeId() == expected_fw_ids);
  std::vector<std::vector<int64_t>> expected_producer_ids{{0}, {4}, {1}, {1},
                                                          {2}, {3}, {2}};
  ASSERT_TRUE(graph.CalcChainRegstId2ProducerChainNodeId()
              == expected_producer_ids);

  std::vector<std::vector<int64_t>> expected_path{
      {0, 1, 2, 5}, {4, 1, 2}, {1, 2}, {1, 2}, {2, 3}, {3, 4}, {2, 5}};
  ASSERT_TRUE(graph.CalcChainRegstId2PathChainNodeIds() == expected_path);

  std::vector<std::vector<int64_t>> expected_splited_regst_ids{
      {0}, {2}, {3}, {6}};

  ASSERT_TRUE(graph.SplitedRegstIds() == expected_splited_regst_ids);

  std::vector<std::vector<int64_t>> expected_cloned_regst_ids{{1}, {4}, {5}};
  ASSERT_TRUE(graph.ClonedRegstIds() == expected_cloned_regst_ids);
}

}  // namespace test

}  // namespace df

}  // namespace oneflow
