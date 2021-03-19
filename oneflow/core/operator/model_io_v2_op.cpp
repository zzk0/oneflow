/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

class ModelInitV2Op : public Operator {
 public:
  void InitFromOpConf() override {
    CHECK(op_conf().has_model_init_v2_conf());
    EnrollInputBn("ref", false)->set_is_mutable(true);
    EnrollOutputBn("out", false);
    EnrollInputBn("tick", false);
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    BlobDesc* out = BlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({parallel_desc.parallel_num()});
    return Maybe<void>::Ok();
  }

  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({1});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferParallelDistributionSignature(
      ParallelDistributionSignature* signature,
      const ParallelDistributionSignature& parallel_distribution_sig_constraints,
      const ParallelDesc& parallel_desc,
      std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
          ParallelDistributionInferHint4Ibn) override {
    (*signature->mutable_bn_in_op2parallel_distribution())["ref"] =
        JUST(ParallelDistributionInferHint4Ibn("ref"))->parallel_distribution();
    const auto& hierarchy = parallel_desc.hierarchy();
    for (int64_t i = 0; i < hierarchy->NumAxes(); ++i) {
      (*signature->mutable_bn_in_op2parallel_distribution())["out"]
          .add_sbp_parallel()
          ->mutable_split_parallel()
          ->set_axis(0);
      (*signature->mutable_bn_in_op2parallel_distribution())["tick"]
          .add_sbp_parallel()
          ->mutable_broadcast_parallel();
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kModelInitV2Conf, ModelInitV2Op);

class ModelLoadV2Op : public Operator {
 public:
  void InitFromOpConf() override {
    CHECK(op_conf().has_model_load_v2_conf());
    EnrollInputBn("path", false);
    EnrollInputBn("ref", false)->set_is_mutable(true);
    EnrollOutputBn("out", false);
    EnrollInputBn("tick", false);
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    BlobDesc* out = BlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({parallel_desc.parallel_num()});
    return Maybe<void>::Ok();
  }

  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({1});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferParallelDistributionSignature(
      ParallelDistributionSignature* signature,
      const ParallelDistributionSignature& parallel_distribution_sig_constraints,
      const ParallelDesc& parallel_desc,
      std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
          ParallelDistributionInferHint4Ibn) override {
    (*signature->mutable_bn_in_op2parallel_distribution())["ref"] =
        JUST(ParallelDistributionInferHint4Ibn("ref"))->parallel_distribution();
    const auto& hierarchy = parallel_desc.hierarchy();
    for (int64_t i = 0; i < hierarchy->NumAxes(); ++i) {
      (*signature->mutable_bn_in_op2parallel_distribution())["out"]
          .add_sbp_parallel()
          ->mutable_split_parallel()
          ->set_axis(0);
      (*signature->mutable_bn_in_op2parallel_distribution())["tick"]
          .add_sbp_parallel()
          ->mutable_broadcast_parallel();
      (*signature->mutable_bn_in_op2parallel_distribution())["path"]
          .add_sbp_parallel()
          ->mutable_broadcast_parallel();
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kModelLoadV2Conf, ModelLoadV2Op);

class ModelSaveV2Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveV2Op);
  ModelSaveV2Op() = default;
  ~ModelSaveV2Op() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_model_save_v2_conf());
    EnrollInputBn("path", false);
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
    EnrollInputBn("tick", false);
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    BlobDesc* out = BlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({parallel_desc.parallel_num()});
    return Maybe<void>::Ok();
  }

  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({1});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferParallelDistributionSignature(
      ParallelDistributionSignature* signature,
      const ParallelDistributionSignature& parallel_distribution_sig_constraints,
      const ParallelDesc& parallel_desc,
      std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
          ParallelDistributionInferHint4Ibn) override {
    (*signature->mutable_bn_in_op2parallel_distribution())["in"] =
        JUST(ParallelDistributionInferHint4Ibn("in"))->parallel_distribution();
    const auto& hierarchy = parallel_desc.hierarchy();
    for (int64_t i = 0; i < hierarchy->NumAxes(); ++i) {
      (*signature->mutable_bn_in_op2parallel_distribution())["out"]
          .add_sbp_parallel()
          ->mutable_split_parallel()
          ->set_axis(0);
      (*signature->mutable_bn_in_op2parallel_distribution())["tick"]
          .add_sbp_parallel()
          ->mutable_broadcast_parallel();
      (*signature->mutable_bn_in_op2parallel_distribution())["path"]
          .add_sbp_parallel()
          ->mutable_broadcast_parallel();
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kModelSaveV2Conf, ModelSaveV2Op);

}  // namespace oneflow
