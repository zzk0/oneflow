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
#ifndef SBP_COLLECTOR_
#define SBP_COLLECTOR_

#include <unordered_map>
#include <vector>
#include <unordered_set>
#include "sbp_constructor.h"
#define DEBUG_COLLECTOR_
using namespace Algorithm;

namespace oneflow {

class SbpCollector {
 public:
  // Stores all the possible SbpParallel.
  std::unordered_map<::oneflow::SbpParallel, int32_t> SbpParallelUniverse;
  // Relationship between id and Sbp Parallel
  vector<::oneflow::SbpParallel> id2SbpParallel;
  // Calculate number of downstream sbp
  std::vector<int32_t> accumulator;
  // A binary set buffer to indicate sets of downstream sbp
  BinarySet bs_buffer;

  SbpCollector() {
    // initialize Sbp Parallel Universe with broadcast.
    SbpParallel sbp_broadcast;
    sbp_broadcast.mutable_broadcast_parallel();
    SbpParallelUniverse[sbp_broadcast] = 0;
  }

  // Collect all the possible Sbp Parallel from a SbpSignature
  void CollectUniverse(SbpSignature& sbp_) {
    ::google::protobuf::Map<::std::string, ::oneflow::SbpParallel>* bn_in_op2sbp_parallels =
        sbp_.mutable_bn_in_op2sbp_parallel();
    for (auto& OpSbpPair : bn_in_op2sbp_parallels) {
      if (SbpParallelUniverse.find(OpSbpPair.second) == SbpParallelUniverse.end()) {
        int32_t curr_size = SbpParallelUniverse.size();
        SbpParallelUniverse[OpSbpPair.second] = curr_size;
        id2SbpParallel.push_back(OpSbpPair.second);
#ifdef DEBUG_COLLECTOR_
        std::cout << curr_size << std::endl;
#endif  // DEBUG_COLLECTOR_
      }
    }
  }
  // Collect all the possible Sbp Parallel from a SbpNode
  void CollectUniverse(SbpNode<SbpSignature>* sbp_node) {
    for (auto& sbp_ : sbp_node->SbpSignatureObjList) { CollectUniverse(sbp_); }
  }
  // Collect all the possible Sbp Parallel from a SbpGraph
  void CollectUniverse(SbpGraph<SbpSignature>& sbp_graph) {
    for (auto* sbp_node : sbp_graph.NodeList) { CollectUniverse(sbp_node); }
    accumulator.resize(SbpParallelUniverse.size(), 0);
    bs_buffer.Initialize(SbpParallelUniverse.size());
  }
  // Export list of possible combination of Sbp Parallels
  vector<BinarySet> ProxySbpCandidate(
      const OpGraph& op_graph,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
      HashMap<std::string, HashMap<LogicalBlobId, Algorithm::SbpNode<SbpSignature>*>>&
          op_name2lbi2sbp_proxy,
      SbpGraph<SbpSignature>& sbp_graph) {
    // mapping from a logical blob id to a group of consumers and corresponding input blob names.
    // mapping from consumers and input blob names to an unordered_set of SBP Parallel.
    HashMap<LogicalBlobId,
            HashMap<std::pair<const OpNode*, std::string>, std::unordered_set<int32_t>>>
        lbi2consumer_bn2sbp_set;
    op_graph.ForEachNode([&](const OpNode* node) {
      OperatorConf::OpTypeCase op_type_case = node->op().op_conf().op_type_case();
      // If not support boxing, just skip it.
      if (IsClassRegistered<int32_t, DisableInputBoxingGroup>(op_type_case)) { return; }
      for (const std::string& ibn : node->op().input_bns()) {
        const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
        Algorithm::SbpNode<SbpSignature>* consumer_sbp_node =
            op_name2sbp_node[node->op().op_name()];
        // a set to store the id of all possible SBP Parallel for a downstream op
        // should filter out B and other repeated SBP Parallel by pre-storing them into an
        // unordered_set
        std::unordered_set<int32_t>& SbpParallelIDs = lbi2consumer_bn2sbp_set[lbi][{node, ibn}];
        for (auto& sbp_sig : consumer_sbp_node->SbpSignatureObjList) {
          const auto& map = sbp_sig.bn_in_op2sbp_parallel();
          const auto& iter = map.find(ibn);
          CHECK_OR_RETURN(iter != map.end())
              << "blob_name " << ibn << " not found in sbp signature";
          const SbpParallel& consumer_sbp = iter->second;
          if (consumer_sbp.has_broadcast_parallel()) continue;
          SbpParallelIDs.insert(SbpParallelUniverse[consumer_sbp]);
        }
      }
    });
    // Decide if we should insert a proxy for each logical blob
    for (const auto& lbi7groups : lbi2consumer_bn2sbp_set) {
      // Only insert proxy for those blobs with multiple downstream consumers.
      if (lbi7groups.second.size() < 2) { continue; }
      const LogicalBlobId& lbi = lbi7groups.first;
      HashMap<std::pair<const OpNode*, std::string>, std::unordered_set<int32_t>>&
          consumer_bn2sbp_set = lbi7groups.second;
      HashMap<std::pair<const OpNode*, std::string>, std::unordered_set<int32_t>>::iterator
          it_begin = consumer_bn2sbp_set.begin();
      const OpNode& producer = it_begin->first.first->ProducerOpNode4Lbi(lbi);
      // store all the binary sets of SBP Parallel into an unordered_set.
      std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidates;
      DFS_SBPset(it_begin, consumer_bn2sbp_set, op_name2sbp_node, ParallelCandidates);
      SbpNode<SbpSignature> *sbp_proxy = sbp_graph.GenerateNode()
      op_name2lbi2sbp_proxy[producer.op().op_name()][lbi] = sbp_proxy;
      
      // Todo: coding
    }
  }

 private:
  // Depth first search to collect Sbp Parallel information for different lbis
  void DFS_SBPset(
      HashMap<std::pair<const OpNode*, std::string>, std::unordered_set<int32_t>>::iterator it,
      HashMap<std::pair<const OpNode*, std::string>, std::unordered_set<int32_t>>&
          consumer_bn2sbp_set,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
      std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidates) {
    if (it == consumer_bn2sbp_set.end()) {
      // store the binary set into an unordered_set
      ParallelCandidates.insert(bs_buffer);
    } else {
      const OpNode* consumer = it->first.first;
      const std::string& ibn = it->first.second;
      Algorithm::SbpNode<SbpSignature>* consumer_sbp_node =
          op_name2sbp_node[consumer->op().op_name()];
      // a set to store the id of all possible SBP Parallel for a downstream op
      // should filter out B and other repeated SBP Parallel by pre-storing them into an
      // unordered_set
      std::unordered_set<int32_t> SbpParallelIDs;
      for (auto& sbp_sig : consumer_sbp_node->SbpSignatureObjList) {
        const auto& map = sbp_sig.bn_in_op2sbp_parallel();
        const auto& iter = map.find(ibn);
        CHECK_OR_RETURN(iter != map.end()) << "blob_name " << ibn << " not found in sbp signature";
        const SbpParallel& consumer_sbp = iter->second;
        if (consumer_sbp.has_broadcast_parallel()) continue;
        SbpParallelIDs.insert(SbpParallelUniverse[consumer_sbp]);
      }
      // next iterator
      HashMap<std::pair<const OpNode*, std::string>, std::unordered_set<int32_t>>::iterator
          it_next = it;
      it_next++;
      // go through all the sbp parallel of different candidates
      for (int32_t SbpParallelNum : SbpParallelIDs) {
        if (++accumulator[SbpParallelNum] == 1) {
          bs_buffer.AddEntry(SbpParallelNum);
          DFS_SBPset(it_next, consumer_bn2sbp_set, op_name2sbp_node);
          bs_buffer.DeleteEntry(SbpParallelNum);
        } else {
          DFS_SBPset(it_next, consumer_bn2sbp_set, op_name2sbp_node);
        }
        accumulator[SbpParallelNum]--;
      }
    }
  }
};  // class SbpCollector

}  // namespace oneflow

#endif  // SBP_COLLECTOR_