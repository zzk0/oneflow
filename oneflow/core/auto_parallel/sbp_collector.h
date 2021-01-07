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
#include <utility>
#include <type_traits>
#include "sbp_constructor.h"
#define DEBUG_COLLECTOR_
using namespace Algorithm;

namespace oneflow {

// compute copy cost
double ComputCopyCostBetweenTwoSbpParallel(const SbpParallel& producer_sbp_parallel,
                                           const SbpParallel& consumer_sbp_parallel,
                                           const BlobDesc& logical_blob_desc,
                                           const ParallelDesc& parallel_desc, bool is_same_sbp);

// Find sbp edge between two given sbp nodes
SbpEdge<SbpSignature>* FindEdgeBetweenNodes(SbpNode<SbpSignature>* sbp_node_producer,
                                            SbpNode<SbpSignature>* sbp_node_consumer);

class SbpCollector {
 public:
  // Stores all the possible SbpParallel.
  std::unordered_map<::oneflow::SbpParallel, int32_t> SbpParallelUniverse;
  // Relationship between id and Sbp Parallel
  std::vector<::oneflow::SbpParallel> id2SbpParallel;
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

  ~SbpCollector() {}

  // Collect all the possible Sbp Parallel from a SbpSignature
  void CollectUniverse(SbpSignature& sbp_) {
    ::google::protobuf::Map<::std::string, ::oneflow::SbpParallel>& bn_in_op2sbp_parallels =
        *sbp_.mutable_bn_in_op2sbp_parallel();
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
  // Initialize sbp proxy with given parallel candidates of a blob
  SbpNode<SbpSignature>* InitializePorxy(
      SbpGraph<SbpSignature>& sbp_graph,
      std::unordered_set<BinarySet, BinarySetHasher>& ParallelCandidates) {
    // Initialize sbp proxy
    SbpNode<SbpSignature>* sbp_proxy = sbp_graph.GenerateNode();
    // move parallel candidates
    for (const BinarySet& parallel_candidate : ParallelCandidates) {
      sbp_proxy->ParallelCandidates.emplace_back(parallel_candidate);
    }
    // Initialize computation cost
    sbp_proxy->Cost.resize(sbp_proxy->ParallelCandidates.size(), 0);
    return sbp_proxy;
  }

  // Initialize copy cost from producer to proxy of producer
  void InitializeCopyCostFromNode2Proxy(SbpNode<SbpSignature>* sbp_proxy,
                                        const LogicalBlobId& lbi) {
    // the only edge from producer  to proxy of producer
    SbpEdge<SbpSignature>* sbp_edge = sbp_proxy->EdgesIn[0];
    SbpNode<SbpSignature>* sbp_node_producer = sbp_edge->StartNode;
    sbp_edge->Cost.resize(sbp_node_producer->SbpSignatureList.size());
    int32_t consumer_sbp_size = sbp_proxy->ParallelCandidates.size();
    // look through sbp signature in producer
    for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_node_producer->SbpSignatureList.size();
         sbp_id_producer++) {
      sbp_edge->Cost[sbp_id_producer].resize(consumer_sbp_size, 0);
    }

    // Assemble copy cost from producer to proxy of producer
    OpNode* producer = sbp_node_producer->op_node;
    // get parallel description. Number of devices.
    const ParallelDesc& parallel_desc = producer->parallel_desc();
    // Need to be careful, the logical blob description should be independent to current
    // SbpParallel. Use producer or op_node?
    const BlobDesc& logical_blob_desc = producer->LogicalBlobDesc4Lbi(lbi);
    const std::string& obn = *CHECK_JUST(producer->op().obn4lbi(lbi));

    // A buffer to store the sbp parallel id
    std::vector<int32_t> sbp_parallel_ids;

    // look through sbp signature in producer
    for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_node_producer->SbpSignatureList.size();
         sbp_id_producer++) {
      // get sbp parallel for a logical blob in producer
      const auto producer_sbp_bn_in_op2sbp_parallel =
          sbp_node_producer->SbpSignatureList[sbp_id_producer]->bn_in_op2sbp_parallel();
      const SbpParallel& sbp_producer = producer_sbp_bn_in_op2sbp_parallel.at(obn);

      // look through sbp parallel set in consumer
      for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
        BinarySet& sbp_parallel_set = sbp_proxy->ParallelCandidates[sbp_id_consumer];
        sbp_parallel_set.QuickOutPut(sbp_parallel_ids);

        // look through all sbp parallels in a sbp parallel set
        for (int32_t sbp_parallel_id : sbp_parallel_ids) {
          // get sbp parallel for a logical blob in consumer
          const SbpParallel& sbp_consumer = id2SbpParallel[sbp_parallel_id];

          // compute copy cost for a specific logical blob
          sbp_edge->Cost[sbp_id_producer][sbp_id_consumer] += ComputCopyCostBetweenTwoSbpParallel(
              sbp_producer, sbp_consumer, logical_blob_desc, parallel_desc, false);
        }
      }
    }
  }

  // Initialize copy cost from proxy of producer to consumers
  void InitializeCopyCostFromProxy2Consumer(
      SbpNode<SbpSignature>* sbp_proxy,
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>&
          consumer_bn2sbp_set,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node) {
    // Connect sbp proxy and consumers
    for (const auto& consumer_bn_group : consumer_bn2sbp_set) {
      // consumer in cost model
      Algorithm::SbpNode<SbpSignature>* sbp_node_consumer =
          op_name2sbp_node[consumer_bn_group.first.first];
      // input blob name of logical blob in consumer
      const std::string& ibn = consumer_bn_group.first.second;

      // check is_mutable in consumer
      OpNode* consumer = sbp_node_consumer->op_node;
      const auto input_blob_modifier_ = consumer->op().InputBlobModifier4Ibn(ibn);
      bool is_same_sbp = input_blob_modifier_.has_is_mutable() && input_blob_modifier_.is_mutable();
      CHECK(!is_same_sbp) << " Create a proxy for an unmutable consumer!\n";

      // Connect sbp proxy and consumer
      sbp_proxy->PointTo(sbp_node_consumer);
      // the sbp edge connecting proxy and consumer
      SbpEdge<SbpSignature>* sbp_edge = FindEdgeBetweenNodes(sbp_proxy, sbp_node_consumer);
      sbp_edge->Cost.resize(sbp_proxy->ParallelCandidates.size());
      int32_t consumer_sbp_size = sbp_node_consumer->SbpSignatureList.size();

      // look through sbp parallel set in proxy
      for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_proxy->ParallelCandidates.size();
           sbp_id_producer++) {
        // initialization for copy cost
        sbp_edge->Cost[sbp_id_producer].resize(consumer_sbp_size, 0);
        // get sbp parallel set for a logical blob in proxy
        Algorithm::BinarySet& parallel_candidate = sbp_proxy->ParallelCandidates[sbp_id_producer];

        // look through sbp signatures in consumers
        for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
          // get sbp parallel for a logical blob in consumer
          const auto consumer_sbp_bn_in_op2sbp_parallel =
              sbp_node_consumer->SbpSignatureList[sbp_id_consumer]->bn_in_op2sbp_parallel();
          const SbpParallel& sbp_consumer = consumer_sbp_bn_in_op2sbp_parallel.at(ibn);

          if (!parallel_candidate.CheckExistency(SbpParallelUniverse[sbp_consumer]))
            sbp_edge->Cost[sbp_id_producer][sbp_id_consumer] = GetMaxVal<float>();
        }
      }
    }
  }

  // Export list of possible combination of Sbp Parallels
  void ProxySbpCandidate(const OpGraph& op_graph,
                         HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
                         SbpGraph<SbpSignature>& sbp_graph) {
    // If needed, we can output the mapping from operator name to its proxy.
    // HashMap<std::string, HashMap<LogicalBlobId, Algorithm::SbpNode<SbpSignature>*>>&
    //     op_name2lbi2sbp_proxy;

    // mapping from a logical blob id to a group of consumers and corresponding input blob names.
    // mapping from consumers and input blob names to an unordered_set of SBP Parallel.
    HashMap<std::pair<std::string, LogicalBlobId>,
            HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>>
        producer_lbi2consumer_bn2sbp_set;
    op_graph.ForEachNode([&](const OpNode* node) {
      OperatorConf::OpTypeCase op_type_case = node->op().op_conf().op_type_case();
      // If not support boxing, just skip it.
      if (IsClassRegistered<int32_t, DisableInputBoxingGroup>(op_type_case)) { return; }
      for (const std::string& ibn : node->op().input_bns()) {
        const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
        const OpNode& producer = node->ProducerOpNode4Lbi(lbi);
        // a set to store the id of all possible SBP Parallel for a downstream op
        // should filter out B and other repeated SBP Parallel by pre-storing them into an
        // unordered_set
        std::unordered_set<int32_t>& SbpParallelIDs = producer_lbi2consumer_bn2sbp_set[{
            producer.op().op_name(), lbi}][{node->op().op_name(), ibn}];
        Algorithm::SbpNode<SbpSignature>* consumer_sbp_node =
            op_name2sbp_node[node->op().op_name()];
        for (auto& sbp_sig : consumer_sbp_node->SbpSignatureObjList) {
          const auto& map = sbp_sig.bn_in_op2sbp_parallel();
          const auto& iter = map.find(ibn);
          CHECK(iter != map.end()) << "blob_name " << ibn << " not found in sbp signature";
          const SbpParallel& consumer_sbp = iter->second;
          // filter out B
          if (consumer_sbp.has_broadcast_parallel()) continue;
          // filter out repeated SBP
          SbpParallelIDs.insert(SbpParallelUniverse[consumer_sbp]);
        }
      }
    });

    // A set of binary set with broadcast only
    std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidatesInitializer;
    BinarySet one_broadcast(SbpParallelUniverse.size());
    one_broadcast.AddEntry(0);
    ParallelCandidatesInitializer.insert(std::move(one_broadcast));

    // Decide if we should insert a proxy for each logical blob
    for (auto& lbi7groups : producer_lbi2consumer_bn2sbp_set) {
      // Only insert proxy for those blobs with multiple downstream consumers.
      if (lbi7groups.second.size() < 2) { continue; }
      const std::string& producer_name = lbi7groups.first.first;
      // producer in cost model
      Algorithm::SbpNode<SbpSignature>* sbp_node_producer = op_name2sbp_node[producer_name];
      const LogicalBlobId& lbi = lbi7groups.first.second;
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>&
          consumer_bn2sbp_set = lbi7groups.second;
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>::iterator it_begin =
          consumer_bn2sbp_set.begin();
      // store all the binary sets of SBP Parallel into an unordered_set.
      std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidates(
          ParallelCandidatesInitializer);
      DFS_SBPset(it_begin, consumer_bn2sbp_set, op_name2sbp_node, ParallelCandidates);
      // Initialize sbp proxy
      SbpNode<SbpSignature>* sbp_proxy = InitializePorxy(sbp_graph, ParallelCandidates);
      // Might be unnecessary
      // op_name2lbi2sbp_proxy[producer_name][lbi] = sbp_proxy;

      // Transfer a logical blob from producer to a sbp proxy of this blob
      sbp_node_producer->PointTo(sbp_proxy);

      // Compute copy cost between producer and proxy
      InitializeCopyCostFromNode2Proxy(sbp_proxy, lbi);

      // Build connection and compute copy cost between proxy and consumers
      InitializeCopyCostFromProxy2Consumer(sbp_proxy, consumer_bn2sbp_set, op_name2sbp_node);

      // Unloading and maybe clipping
      for (const auto& consumer_bn_group : consumer_bn2sbp_set) {
        // consumer in cost model
        Algorithm::SbpNode<SbpSignature>* sbp_node_consumer =
            op_name2sbp_node[consumer_bn_group.first.first];
        // the sbp edge connecting producer and consumer
        SbpEdge<SbpSignature>* edge_found =
            FindEdgeBetweenNodes(sbp_node_producer, sbp_node_consumer);
        // unload logical blob from sbp edges
        edge_found->UnloadLbi(lbi);
        // clip this edge if it no longer carrys any blob
        if (edge_found->EmptyLbi()) sbp_graph.ClipEdge(edge_found);
      }

      // Todo: coding
    }
  }

 private:
  // Depth first search to collect Sbp Parallel information for different lbis
  void DFS_SBPset(
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>::iterator it,
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>&
          consumer_bn2sbp_set,
      HashMap<std::string, Algorithm::SbpNode<SbpSignature>*>& op_name2sbp_node,
      std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidates) {
    if (it == consumer_bn2sbp_set.end()) {
      // store the binary set into an unordered_set
      ParallelCandidates.insert(bs_buffer);
    } else {
      const std::string& consumer_name = it->first.first;
      const std::string& ibn = it->first.second;
      Algorithm::SbpNode<SbpSignature>* consumer_sbp_node = op_name2sbp_node[consumer_name];
      // a set to store the id of all possible SBP Parallel for a downstream op
      // should filter out B and other repeated SBP Parallel by pre-storing them into an
      // unordered_set
      std::unordered_set<int32_t> SbpParallelIDs;
      for (auto& sbp_sig : consumer_sbp_node->SbpSignatureObjList) {
        const auto& map = sbp_sig.bn_in_op2sbp_parallel();
        const auto& iter = map.find(ibn);
        CHECK(iter != map.end()) << "blob_name " << ibn << " not found in sbp signature";
        const SbpParallel& consumer_sbp = iter->second;
        if (consumer_sbp.has_broadcast_parallel()) continue;
        SbpParallelIDs.insert(SbpParallelUniverse[consumer_sbp]);
      }
      // next iterator
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>::iterator it_next =
          it;
      it_next++;
      // go through all the sbp parallel of different candidates
      for (int32_t SbpParallelNum : SbpParallelIDs) {
        if (++accumulator[SbpParallelNum] == 1) {
          bs_buffer.AddEntry(SbpParallelNum);
          DFS_SBPset(it_next, consumer_bn2sbp_set, op_name2sbp_node, ParallelCandidates);
          bs_buffer.DeleteEntry(SbpParallelNum);
        } else {
          DFS_SBPset(it_next, consumer_bn2sbp_set, op_name2sbp_node, ParallelCandidates);
        }
        accumulator[SbpParallelNum]--;
      }
    }
  }
};  // class SbpCollector

}  // namespace oneflow

#endif  // SBP_COLLECTOR_