#include "oneflow/core/auto_placement/df_func.h"
#include "oneflow/core/auto_placement/demo_chain_graph.h"

namespace oneflow {

namespace df {

namespace {

Tensor CalcTaskNodeComputeTime(const Tensor& chain_node_placement) {
  Tensor row_ones(Shape({chain_node_placement.shape().At(0)}), 1);
  auto placement_copies = Clone(chain_node_placement, 2);
  Tensor col = MatrixColSum(placement_copies.at(0));
  Tensor col_sum = TensorProduct(row_ones, col);
  return ElemWiseDiv(placement_copies.at(1), col_sum);
}

Tensor CalcDeviceComputeTime(const Tensor& chain_node_placement) {
  return MatrixRowSum(CalcTaskNodeComputeTime(chain_node_placement));
}

Tensor CalcTaskNodeTime(const Tensor& chain_node_placement) {
  Tensor compute_time = CalcTaskNodeComputeTime(chain_node_placement);
  Tensor col_ones(Shape({chain_node_placement.shape().At(1)}), 1);
  auto compute_time_copies = Clone(compute_time, 2);
  Tensor row_sum =
      TensorProduct(MatrixRowSum(compute_time_copies.at(0)), col_ones);
  return Mul(Tensor(0.5), Add(row_sum, compute_time_copies.at(1)));
}

Tensor CalcRegstDuration(const Tensor& chain_node_placement,
                         const DemoChainGraph& chain_graph) {
  Tensor task_node_time = CalcTaskNodeTime(chain_node_placement);
  Tensor chain_node_time = MatrixColMax(task_node_time);
  auto GetTime = [chain_node_time](int64_t chain_node_id) -> double {
    return chain_node_time.At(chain_node_id);
  };
  auto regst2path = chain_graph.CalcChainRegstId2PathChainNodeIds(GetTime);
  return ColIndexReduce(task_node_time, regst2path);
}

Tensor CalcRegstMemory(const Tensor& chain_node_placement,
                       const DemoChainGraph& chain_graph) {
  auto regst2producer = chain_graph.CalcChainRegstId2ProducerChainNodeId();
  int64_t regst_num = regst2producer.size();
  Tensor regst_placement = ColIndexReduce(chain_node_placement, regst2producer);
  Tensor row_ones(Shape({regst_placement.shape().At(0)}), 1);
  auto copies = Clone(regst_placement, 3);
  Tensor col_sum = TensorProduct(row_ones, MatrixColSum(copies.at(0)));
  Tensor split_workload_ratio = ElemWiseDiv(copies.at(1), col_sum);
  Tensor clone_workload_ratio = Tanh(copies.at(2));
  Tensor clone_weight = TensorProduct(
      row_ones, Tensor(Shape({regst_num}), chain_graph.RegstId2IsCloned()));
  return Add(ElemWiseMul(clone_workload_ratio, clone_weight),
             ElemWiseMul(split_workload_ratio, Sub(Tensor(1), clone_weight)));
}

Tensor CalcIIRatio(const Tensor& chain_node_placement,
                   const DemoChainGraph& chain_graph, int piece_num_in_batch) {
  auto ii_ratios = chain_graph.RegstIIRatio(piece_num_in_batch);
  int64_t regst_num = ii_ratios.size();
  Tensor ii_ratio_tensor(Shape({regst_num}), ii_ratios);
  Tensor row_ones(Shape({chain_node_placement.shape().At(0)}), 1);
  return Reciprocal(TensorProduct(row_ones, ii_ratio_tensor));
}

Tensor CalcDeviceMemII(const Tensor& chain_node_placement,
                       const DemoChainGraph& chain_graph,
                       int piece_num_in_batch, double mem_size_per_device) {
  Tensor regst_mem = CalcRegstMemory(chain_node_placement, chain_graph);
  Tensor regst_duration = CalcRegstDuration(chain_node_placement, chain_graph);
  Tensor ii_ratio =
      CalcIIRatio(chain_node_placement, chain_graph, piece_num_in_batch);
  auto regst_mem_copies = Clone(regst_mem, 2);
  Tensor weighted_mem_time = ElemWiseMul(ElemWiseMul(ii_ratio, regst_duration),
                                         regst_mem_copies.at(0));
  Tensor weighted_mem_ceil_diff =
      ElemWiseMul(Sub(Tensor(1.5), ii_ratio), regst_mem_copies.at(1));
  Tensor device_mem_time = MatrixRowSum(weighted_mem_time);
  Tensor device_mem =
      Sub(Tensor(mem_size_per_device), MatrixRowSum(weighted_mem_ceil_diff));
  Tensor epsilon(0.000000000001);
  Tensor row_ones(Shape({chain_node_placement.shape().At(0)}), 1);
  Tensor cliped_device_mem = Max(device_mem, TensorProduct(row_ones, epsilon));
  return ElemWiseDiv(device_mem_time, cliped_device_mem);
}

void AutoPlacementMemoryDemo() {
  DemoChainGraph chain_graph([](DemoChainGraphBuilder* builder) {
    builder->Backward(builder->Op(
        "loss",
        {builder->ModelOp(
            "op3", {builder->ModelOp(
                       "op2", {builder->ModelOp(
                                  "op1", {builder->ModelOp("op0")})})})}));
  });
  int64_t fw_node_num = chain_graph.FwChainNodeNum();
  // std::cout << fw_node_num << std::endl;
  // return;
  Shape shape({7, fw_node_num});
  Tensor fw_var(shape, [](size_t index) { return index % 2 ? 1 : 0; });
  Tensor epsilon(0.000000000001);
  Tensor ceil_tensor(shape, 1);
  Tensor floor_tensor(shape, 0.000000000001);
  FOR_RANGE(int, i, 0, 50000) {
    double lr = 0.01;
    Tensor x = Add(Square((FixedExpectation(Update(&fw_var, lr), 1))), epsilon);
    const auto& placement_copies = Clone(x, 3);
    Tensor computation_ii = CalcDeviceComputeTime(placement_copies.at(0));
    Tensor ii = MaxElem(computation_ii);
    Tensor penalty = Add(Variance(MatrixColMax(placement_copies.at(1))),
                         DoubleVariance(placement_copies.at(2)));
    Backward(Add(ii, penalty));

    std::cout << "x: ";
    for (double i : x.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "computation_ii: ";
    for (double i : computation_ii.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl << std::endl;

    //    Backward(Variance(MatrixColMax(Update(&var, lr))));
    //    std::cout << "var: ";
    //    for (double i : var.buffer().data()) { std::cout << i << " "; }
    //    std::cout << std::endl;
  }
}

void AutoPlacementComputationDemo() {
  Tensor var(Shape({4, 5}), [](size_t index) { return index % 2 ? 0 : 1; });
  Tensor row_ones(Shape({var.shape().At(0)}), 1);
  Tensor col_ones(Shape({var.shape().At(1)}), 1);
  Tensor epsilon(0.000000001);
  FOR_RANGE(int, i, 0, 10000) {
    double lr = 0.001;

    Tensor x = Add(Square(FixedExpectation(Update(&var, lr), 1)), epsilon);
    const auto& x_copies = Clone(x, 4);
    Tensor row = MatrixRowSum(x_copies.at(0));
    Tensor col = MatrixColSum(x_copies.at(1));
    Tensor load = ElemWiseDiv(x_copies.at(2), TensorProduct(row_ones, col));
    Tensor table = ElemWiseMul(TensorProduct(row, col_ones), load);
    Tensor ii = MaxElem(table);
    Backward(Add(ii, Variance(MatrixColMax(x_copies.at(3)))));

    std::cout << "x: ";
    for (double i : x.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "row: ";
    for (double i : row.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "col: ";
    for (double i : col.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "table: ";
    for (double i : table.buffer().data()) { std::cout << i << " "; }
    std::cout << std::endl << std::endl;

    //    Backward(Variance(MatrixColMax(Update(&var, lr))));
    //    std::cout << "var: ";
    //    for (double i : var.buffer().data()) { std::cout << i << " "; }
    //    std::cout << std::endl;
  }
}

void DifferentialDemo() {
  // AutoPlacementComputationDemo();
  AutoPlacementMemoryDemo();
}

}  // namespace

}  // namespace df

}  // namespace oneflow

int main(int argc, char** argv) {
  oneflow::df::DifferentialDemo();
  return 0;
}
