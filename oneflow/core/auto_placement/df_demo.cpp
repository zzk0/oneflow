#include <random>
#include <cmath>
#include "oneflow/core/auto_placement/df_func.h"
#include "oneflow/core/auto_placement/demo_chain_graph.h"

namespace oneflow {

namespace df {

namespace {

Tensor CalcTaskNodeComputeTime(const Tensor& chain_node_placement) {
  return chain_node_placement;
}

Tensor CalcDeviceComputeTime(const Tensor& prob_matrix) {
  return MatrixRowSum(prob_matrix);
}

Tensor CalcTaskNodeTime(const Tensor& chain_node_placement) {
  return chain_node_placement;
}

Tensor CalcRegstDuration(const Tensor& chain_node_placement,
                         const DemoChainGraph& chain_graph) {
  Tensor row_ones(Shape({chain_node_placement.shape().At(0)}), 1);
  Tensor task_node_time = CalcTaskNodeTime(chain_node_placement);
  Tensor chain_node_time = MatrixColSum(task_node_time);
  const auto& regst2path = chain_graph.chain_regst_id2path_chain_node_ids();
  return ColIndexReduce(TensorProduct(row_ones, chain_node_time), regst2path);
}

Tensor CalcRegstMemory(const Tensor& chain_node_placement,
                       const DemoChainGraph& chain_graph) {
  const auto& regst2producer =
      chain_graph.chain_regst_id2producer_chain_node_id();
  int64_t regst_num = regst2producer.size();
  Tensor regst_placement = ColIndexReduce(chain_node_placement, regst2producer);
  Tensor row_ones(Shape({regst_placement.shape().At(0)}), 1);
  auto copies = Clone(regst_placement, 3);
  Tensor col_sum = TensorProduct(row_ones, MatrixColSum(copies.at(0)));
  Tensor split_workload_ratio = ElemWiseDiv(copies.at(1), col_sum);
  Tensor clone_workload_ratio = copies.at(2);
  Tensor clone_weight = TensorProduct(
      row_ones,
      Tensor(Shape({regst_num}), chain_graph.chain_regst_id2is_cloned()));
  auto clone_weight_copies = Clone(clone_weight, 2);
  return ADD(ElemWiseMul(clone_workload_ratio, clone_weight_copies.at(0)),
             ElemWiseMul(split_workload_ratio,
                         Sub(Tensor(1), clone_weight_copies.at(1))));
}

Tensor CalcIIRatio(const Tensor& chain_node_placement,
                   const DemoChainGraph& chain_graph) {
  const auto& ii_ratios = chain_graph.chain_regst_id2ii_scale();
  int64_t regst_num = ii_ratios.size();
  Tensor ii_ratio_tensor(Shape({regst_num}), ii_ratios);
  Tensor row_ones(Shape({chain_node_placement.shape().At(0)}), 1);
  return Reciprocal(TensorProduct(row_ones, ii_ratio_tensor));
}

Tensor CalcDeviceMemBasicConsumed(const Tensor& chain_node_placement,
                                  Tensor regst_duration,
                                  const DemoChainGraph& chain_graph) {
  Tensor regst_mem = CalcRegstMemory(chain_node_placement, chain_graph);
  Tensor ii_ratio = CalcIIRatio(chain_node_placement, chain_graph);
  return MatrixRowSum(
      ElemWiseMul(ElemWiseMul(ii_ratio, regst_duration), regst_mem));
}

Tensor CalcDeviceCopiedRegstMem(const Tensor& chain_node_prob,
                                Tensor regst_duration,
                                const DemoChainGraph& chain_graph) {
  auto chain_node_prob_copies = Clone(chain_node_prob, 2);
  Tensor edge_src_prob = ColIndexReduce(
      chain_node_prob_copies.at(0), chain_graph.edge_id2src_chain_node_id());
  Tensor edge_dst_prob = ColIndexReduce(
      chain_node_prob_copies.at(1), chain_graph.edge_id2dst_chain_node_id());
  Tensor edge_prob = Mul(Tensor(0.5), Abs(Sub(edge_src_prob, edge_dst_prob)));
  Tensor edge_regst_duration_prob =
      ColIndexReduce(regst_duration, chain_graph.edge_id2chain_regst_id());
  Tensor copied_task_regst_prob =
      ElemWiseMul(edge_prob, edge_regst_duration_prob);
  return MatrixRowSum(copied_task_regst_prob);
}

Tensor CalcDeviceCopiedRegstMem(const Tensor& chain_node_prob,
                                const DemoChainGraph& chain_graph) {
  auto chain_node_prob_copies = Clone(chain_node_prob, 2);
  Tensor regst_duration =
      CalcRegstDuration(chain_node_prob_copies.at(0), chain_graph);
  return CalcDeviceCopiedRegstMem(chain_node_prob_copies.at(1), regst_duration,
                                  chain_graph);
}

Tensor CalcDeviceMemConsumed(const Tensor& chain_node_prob,
                             const DemoChainGraph& chain_graph) {
  auto chain_node_prob_copies = Clone(chain_node_prob, 3);
  Tensor regst_duration =
      CalcRegstDuration(chain_node_prob_copies.at(2), chain_graph);
  auto regst_duration_copies = Clone(regst_duration, 2);
  return ADD(
      CalcDeviceMemBasicConsumed(chain_node_prob_copies.at(0),
                                 regst_duration_copies.at(0), chain_graph),
      CalcDeviceCopiedRegstMem(chain_node_prob_copies.at(1),
                               regst_duration_copies.at(1), chain_graph));
}

Tensor CalcDeviceMemII(const Tensor& chain_node_placement,
                       const DemoChainGraph& chain_graph,
                       double mem_size_per_device) {
  auto placement_copies = Clone(chain_node_placement, 2);
  Tensor regst_mem = CalcRegstMemory(placement_copies.at(0), chain_graph);
  Tensor regst_duration =
      CalcRegstDuration(placement_copies.at(1), chain_graph);
  Tensor ii_ratio = CalcIIRatio(chain_node_placement, chain_graph);
  auto ii_ratio_copies = Clone(ii_ratio, 2);
  auto regst_mem_copies = Clone(regst_mem, 2);
  Tensor weighted_mem_time =
      ElemWiseMul(ElemWiseMul(ii_ratio_copies.at(0), regst_duration),
                  regst_mem_copies.at(0));
  Tensor weighted_mem_ceil_diff = ElemWiseMul(
      Sub(Tensor(1.5), ii_ratio_copies.at(1)), regst_mem_copies.at(1));
  Tensor device_mem_time = MatrixRowSum(weighted_mem_time);
  Tensor device_mem =
      Sub(Tensor(mem_size_per_device), MatrixRowSum(weighted_mem_ceil_diff));
  int64_t dev_num = chain_node_placement.shape().At(0);
  Tensor row_ones(Shape({dev_num}), 1);
  Tensor epsilon = Reshape(TensorProduct(row_ones, Tensor(0.000000000001)),
                           Shape({dev_num}));
  Tensor cliped_device_mem = Max(device_mem, epsilon);
  return ElemWiseDiv(device_mem_time, cliped_device_mem);
}

Tensor ProbabilityMatrix(Tensor* var, double lr) {
  Tensor row_ones(Shape({var->shape().At(0)}), 1);
  Tensor epsilon(0.000000000000000001);
  Tensor x = ADD(Square(FixedExpectation(Update(var, lr), 1)), epsilon);
  auto x_copies = Clone(x, 2);
  Tensor x_col_sum = TensorProduct(row_ones, MatrixColSum(x_copies.at(0)));
  return ElemWiseDiv(x_copies.at(1), x_col_sum);
}

std::function<double()> MakeFlation(int keep, double ratio) {
  std::shared_ptr<int> exec_cnt(new int(-1));
  return [=]() {
    if (++(*exec_cnt) < keep) { return 1.0; }
    return 1.0 / (((*exec_cnt) - keep) * ratio + 1.0);
  };
}

std::function<double()> MakeFlation(int keep) {
  return MakeFlation(keep, 0.005);
}

void AutoPlacementMemoryDemo() {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> distr(1, 0.01);
  DemoChainGraph chain_graph(4, [](DemoChainGraphBuilder* builder) {
    auto regst = builder->ModelOp("op0");
    FOR_RANGE(int, i, 1, 63) {
      regst = builder->ModelOp("op" + std::to_string(i), {regst});
    }
    builder->Backward(builder->ModelOp("loss", {regst}));
  });
  const auto& chain_node_id2fw_id =
      chain_graph.chain_node_id2fw_chain_node_id();
  int64_t fw_node_num = chain_graph.FwChainNodeNum();
  Shape shape({2, fw_node_num});
  Tensor fw_var(shape, [&](size_t index) { return distr(gen); });
  Tensor fw_prob;
  const auto& chain_node_id2name = chain_graph.chain_node_id2chain_node_name();
  double bugo = 2;
  double rethink_threshold = 20;
  Tensor decision_ratio(Shape({fw_node_num}), [&](int64_t index) {
    return 1 + fw_node_num * 0.5 / (index + 1);
  });
  std::function<double()> MemFlation = MakeFlation(100);
  FOR_RANGE(int, step, 0, 100000) {
    double lr = 0.01;
    if (step % (static_cast<int>(bugo += 0.05))) {
      fw_prob = ProbabilityMatrix(&fw_var, lr);
      auto fw_prob_copies = Clone(fw_prob, 2);
      Tensor chain_node_prob =
          ColIndexReduce(fw_prob_copies.at(0), chain_node_id2fw_id);
      auto chain_prob_copies = Clone(chain_node_prob, 2);
      Tensor computation_ii = MatrixRowSum(chain_prob_copies.at(0));
      Tensor dev_mem =
          CalcDeviceMemConsumed(chain_prob_copies.at(1), chain_graph);
      Tensor normalized_dev_mem =
          Mul(Tensor(2.5 * MemFlation()), Sqrt(dev_mem));
      Tensor fw_indecision =
          Mul(Sub(MatrixColSum(Sqrt(fw_prob_copies.at(1))), Tensor(1)),
              decision_ratio);
      Tensor indecision = Sum(fw_indecision);
      Tensor balance = ADD(indecision, ADD(AvgAbsDeviation(normalized_dev_mem),
                                           AvgAbsDeviation(computation_ii)));
      BackwardRun(balance);

      if (step % 10 == 0) {
        std::cout << "fw_prob: " << std::endl;
        FOR_RANGE(int, j, 0, fw_prob.shape().At(1)) {
          FOR_RANGE(int, i, 0, fw_prob.shape().At(0)) {
            double x = fw_prob.At(i, j);
            if (x < 0.01) { x = 0; }
            if (x > 0.99) { x = 1; }
            std::cout << x << "\t";
          }
          std::cout << std::endl;
        }
        std::cout << "indecision: " << indecision.At(0) << std::endl;
        std::cout << "computation_ii: ";
        for (double i : computation_ii.buffer().data()) {
          std::cout << i << " ";
        }
        std::cout << std::endl;
        std::cout << "normalized_dev_mem: ";
        for (double i : normalized_dev_mem.buffer().data()) {
          std::cout << i << " ";
        }
        std::cout << std::endl;

        std::vector<int64_t> fw_id2dev_id(fw_prob.shape().At(1));
        FOR_RANGE(int, j, 0, fw_prob.shape().At(1)) {
          double max_val = 0;
          int max_index = 0;
          FOR_RANGE(int, i, 0, fw_prob.shape().At(0)) {
            if (max_val < fw_prob.At(i, j)) {
              max_val = fw_prob.At(i, j);
              max_index = i;
            }
          }
          fw_id2dev_id.at(j) = max_index;
        }
        std::vector<std::list<int64_t>> dev_id2fw_ids(fw_prob.shape().At(0));
        FOR_RANGE(int, fw_id, 0, fw_id2dev_id.size()) {
          dev_id2fw_ids.at(fw_id2dev_id.at(fw_id)).push_back(fw_id);
        }

        FOR_RANGE(int, dev_id, 0, dev_id2fw_ids.size()) {
          std::cout << "device " << dev_id << ": ";
          for (int64_t fw_id : dev_id2fw_ids.at(dev_id)) {
            std::cout << chain_node_id2name.at(fw_id) << " ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
      if (indecision.At(0) < rethink_threshold) {
        MemFlation = MakeFlation(100);
        rethink_threshold -= 1;
        const auto& edge_id2src_id = chain_graph.edge_id2src_chain_node_id();
        const auto& edge_id2dst_id = chain_graph.edge_id2dst_chain_node_id();
        auto old_fw_var = fw_var.buffer();
        FOR_RANGE(int, conv_iter, 0, 1) {
          chain_node_prob = ColIndexReduce(fw_prob, chain_node_id2fw_id);
          Tensor edge_src_prob =
              ColIndexReduce(chain_node_prob, edge_id2src_id);
          Tensor edge_dst_prob =
              ColIndexReduce(chain_node_prob, edge_id2dst_id);
          Tensor edge_prob =
              Mul(Tensor(0.5), Abs(Sub(edge_src_prob, edge_dst_prob)));
          FOR_RANGE(int, i, 0, edge_prob.shape().At(0)) {
            FOR_RANGE(int, j, 0, edge_prob.shape().At(1)) {
              if (edge_prob.At(i, j) > 0.2) {
                int64_t src_fw_id =
                    chain_node_id2fw_id.at(edge_id2src_id.at(j).at(0)).at(0);
                int64_t dst_fw_id =
                    chain_node_id2fw_id.at(edge_id2dst_id.at(j).at(0)).at(0);
                double avg =
                    (old_fw_var.At(i, src_fw_id) + old_fw_var.At(i, dst_fw_id))
                    / 2;
                fw_var.At(i, src_fw_id) = avg;
                fw_var.At(i, dst_fw_id) = avg;
              }
            }
          }
        }
        bugo = 20;
      }
    } else {
      FOR_RANGE(int, i, 0, 3) {
        fw_prob = ProbabilityMatrix(&fw_var, lr);
        Tensor chain_node_prob = ColIndexReduce(fw_prob, chain_node_id2fw_id);
        Tensor copied_mem =
            Sum(CalcDeviceCopiedRegstMem(chain_node_prob, chain_graph));
        BackwardRun(copied_mem);
      }
    }
  }
}

void AutoPlacementComputationDemo() {
  Tensor var(Shape({4, 5}), [](size_t index) { return index % 2 ? 0 : 1; });
  Tensor row_ones(Shape({var.shape().At(0)}), 1);
  Tensor col_ones(Shape({var.shape().At(1)}), 1);
  Tensor epsilon(0.000000001);
  FOR_RANGE(int, i, 0, 10000) {
    double lr = 0.001;

    Tensor x = ADD(Square(FixedExpectation(Update(&var, lr), 1)), epsilon);
    const auto& x_copies = Clone(x, 4);
    Tensor row = MatrixRowSum(x_copies.at(0));
    Tensor col = MatrixColSum(x_copies.at(1));
    Tensor load = ElemWiseDiv(x_copies.at(2), TensorProduct(row_ones, col));
    Tensor table = ElemWiseMul(TensorProduct(row, col_ones), load);
    Tensor ii = MaxElem(table);
    BackwardRun(ADD(ii, Variance(MatrixColMax(x_copies.at(3)))));

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
