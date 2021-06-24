/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#define OV_TF_FRONTEND 0

#include <iomanip>
#include <chrono>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"

#include "api.h"
#include "logging/ovtf_log.h"
#include "logging/tf_graph_writer.h"
#include "openvino_tensorflow/assign_clusters.h"
#include "openvino_tensorflow/backend_manager.h"
#include "openvino_tensorflow/cluster_manager.h"
#include "openvino_tensorflow/deassign_clusters.h"
#include "openvino_tensorflow/encapsulate_clusters.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_utils.h"

#if OV_TF_FRONTEND
#include "ngraph/frontend/tensorflow/tensorflow_frontend/tensorflow.hpp"
#include "tensorflow/core/graph/algorithm.h"
#include <ie_core.hpp>
#else
#include "ocm/include/ocm_nodes_checker.h"
#endif

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

class NGraphRewritePass : public GraphOptimizationPass {
 public:
  virtual Status Run(const GraphOptimizationPassOptions& options) = 0;

 protected:
  // Returns a fresh "serial number" to avoid filename collisions in the graph
  // dumps.
  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

int NGraphRewritePass::s_serial_counter = 0;
mutex NGraphRewritePass::s_serial_counter_mutex;

//
// Pass that rewrites the graph for nGraph operation.
//
// The pass has several phases, each executed in the below sequence:
//
//   1. Marking [mark_for_clustering.cc]
//   2. Cluster Assignment [assign_clusters.cc]
//   3. Cluster Deassignment [deassign_clusters.cc]
//   4. Cluster Encapsulation [encapsulate_clusters.cc]

class NGraphEncapsulationPass : public NGraphRewritePass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    // If we don't get a main graph, log that fact and bail.
    if (options.graph == nullptr) {
      OVTF_VLOG(0) << "NGraphEncapsulationPass: options.graph == nullptr";
      return Status::OK();
    }

    // For filename generation purposes, grab a fresh index. This is just an
    // arbitrary integer to avoid filename collisions resulting from subsequent
    // runs of this pass.
    int idx = FreshIndex();

    tensorflow::Graph* graph = options.graph->get();
    // If requested, dump unmarked graphs.
    util::DumpTFGraph(graph, idx, "unmarked");

    // If ngraph is disabled via openvino_tensorflow api or OPENVINO_TF_DISABLE
    // is set
    // we will not do anything; all subsequent
    // passes become a no-op.
    bool ovtf_not_enabled =
        (!api::IsEnabled()) || (std::getenv("OPENVINO_TF_DISABLE") != nullptr);
    bool already_processed = util::IsAlreadyProcessed(graph);
    if (!already_processed && ovtf_not_enabled) {
      OVTF_VLOG(0) << "NGraph is available but disabled.";
    }
    if (ovtf_not_enabled || already_processed) {
      OVTF_VLOG(1) << std::string("Rewrite pass will not run because ") +
                          (already_processed ? "graph is already preprocessed"
                                             : "ngraph is disabled");
      NGraphClusterManager::EvictAllClusters();
      return Status::OK();
    }

    // Now Process the Graph

    // 1. Mark for clustering then, if requested, dump the graphs.
    std::set<string> skip_these_nodes = {};

    // OCM call for marking supported nodes
    std::string device;
    BackendManager::GetBackendName(device);
    const char* device_id(device.c_str());
    std::string ov_version;
#if defined(OPENVINO_2021_2)
    ov_version = "2021.2";
#else if defined(OPENVINO_2021_3)
    ov_version = "2021.3";
#endif

      using namespace std::chrono;
      auto caps_start = high_resolution_clock::now();

#if OV_TF_FRONTEND

    // Phase 1: partially convert function with FE

      std::shared_ptr<ngraph::Function> ng_function;
      try {
          std::cerr << "slyalin: FE as OCM: started frontend phase\n";
          ngraph::frontend::FrontEndTensorflow frontend;

          auto start = high_resolution_clock::now();
#if 1
          vector<Node*> sorted;
          GetReversePostOrder(*options.graph->get(), &sorted, NodeComparatorName());
          std::cout << "[ INFO ] Graph was sorted!\n";

          vector<std::shared_ptr<NodeDef>> gd(sorted.size());
          for(size_t i = 0; i < gd.size(); ++i)
          {
              // TODO: Use Node directly instead of NodeDef, because some attributes are not reliable according to
              // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/graph/graph.h#L97
              gd[i] = std::make_shared<NodeDef>(sorted[i]->def());
          }
#else
          auto gd = std::make_shared<::tensorflow::GraphDef>();
          options.graph->get()->ToGraphDef(gd.get());
#endif

          auto stop = high_resolution_clock::now();
          auto duration = duration_cast<milliseconds>(stop - start);
          cout << "ToGraphDef preparation took: " << duration.count() << " ms" << endl;
          auto inputModelTensorflow = std::make_shared<ngraph::frontend::InputModelTensorflow>(gd);
          ng_function = frontend.convert(inputModelTensorflow);
          std::cerr << "ng_function->get_results().size()" << ng_function->get_results().size() << "\n";
          std::cerr << "slyalin: FE as OCM: finished frontend phase\n";
      }
      catch(...)
      {
          std::cerr << "Excpetion thrown when converting TF graph def with frontend\n";
      }

    // TODO

    InferenceEngine::Core core;
    core.QueryNetwork(InferenceEngine::CNNNetwork(ng_function), device);


    // TODO

    // Phase 3: put data in the format compatible with the remaining code

      std::vector<void*> nodes_list;

#else

    ocm::Framework_Names fName = ocm::Framework_Names::TF;
    ocm::FrameworkNodesChecker FC(fName, device_id, ov_version,
                                  options.graph->get());
    std::set<std::string> disabled_ops_set = api::GetDisabledOps();
    if (device == "HDDL" && std::getenv("OPENVINO_TF_ENABLE_BATCHING")) {
      std::vector<std::string> batched_disabled_ops = {"Shape"};
      for (int i = 0; i < batched_disabled_ops.size(); i++) {
        disabled_ops_set.insert(batched_disabled_ops[i]);
      }
    }
    FC.SetDisabledOps(disabled_ops_set);
    std::vector<void*> nodes_list = FC.MarkSupportedNodes();
#endif

      auto caps_stop = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(caps_stop - caps_start);
      cout << "OCM part took: " << duration.count() << " ms" << std::endl;

    // cast back the nodes in the TF format and mark the nodes for clustering
    // (moved out from MarkForClustering function)
    const std::map<std::string, SetAttributesFunction>& set_attributes_map =
        GetAttributeSetters();
    for (auto void_node : nodes_list) {
      // TODO(amprocte): move attr name to a constant
      tensorflow::Node* node = (tensorflow::Node*)void_node;
      node->AddAttr("_ovtf_marked_for_clustering", true);
      auto it = set_attributes_map.find(node->type_string());
      if (it != set_attributes_map.end()) {
        it->second(node);
      }
    }

    util::DumpTFGraph(graph, idx, "marked");

    // 2. Assign clusters then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(AssignClusters(graph));
    util::DumpTFGraph(graph, idx, "clustered");

    // 3. Deassign trivial clusters then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(DeassignClusters(graph));
    util::DumpTFGraph(graph, idx, "declustered");

    // 4. Encapsulate clusters then, if requested, dump the graphs.
    std::unordered_map<std::string, std::string> config_map;
    auto status = EncapsulateClusters(graph, idx, config_map);
    if (status != Status::OK()) {
      return status;
    }

    util::DumpTFGraph(graph, idx, "encapsulated");
    return Status::OK();
  }
};

}  // namespace openvino_tensorflow

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      openvino_tensorflow::NGraphEncapsulationPass);
}  // namespace tensorflow
