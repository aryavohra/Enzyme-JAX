#include "src/enzyme_ad/jax/Passes/target/cxxbridge/tensat/src/input.rs.h"
#include "src/enzyme_ad/jax/Passes/target/cxxbridge/rust/cxx.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace rust::cxxbridge1;

namespace {
class EqualitySaturationPass : public PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "equality-saturation-pass"; }
  StringRef getDescription() const override { return "Optimizes HLO graph using a Rust-based optimizer"; }

  int measureCost(Operation *op) {
    return 0;
  }
    
  Box<TensorInfo> dfs(
      Operation *op,
      std::unordered_map<Operation*, Box<TensorInfo>> *opToTensorInfo,
      Box<CppGraphConverter> &graph
      ) {
    if (opToTensorInfo->find(op) != opToTensorInfo->end()) {
        return std::move(opToTensorInfo->at(op));
    }

    Box<TensorInfo> tensorInfo = Box<TensorInfo>::from_raw(nullptr);
    string opName = op->getName().getStringRef().str();

    if (opName == "func.func") {
    }
    else if (isa<stablehlo::ConstantOp>(op)) {
        auto constantOp = cast<stablehlo::ConstantOp>(op);
        tensorInfo = graph->new_constant_op(0, measureCost(op));
    }
    else if (isa<stablehlo::MulOp>(op)) {
        auto mulOp = cast<stablehlo::MulOp>(op);
        auto lhs = mulOp.getLhs().getDefiningOp();
        auto rhs = mulOp.getRhs().getDefiningOp();
        tensorInfo = graph->new_mul_op(*dfs(lhs, opToTensorInfo, graph), *dfs(rhs, opToTensorInfo, graph), measureCost(op));
    }

    if (tensorInfo.into_raw() != nullptr) {
        opToTensorInfo->insert({op, std::move(tensorInfo)});
        return std::move(opToTensorInfo->at(op));
    }
  }
  
  void dfsTraverse(Operation *op, std::unordered_map<Operation*, Box<TensorInfo>> *opToTensorInfo, Box<CppGraphConverter> &graph, std::unordered_map<int, Operation*> *unsupportedOpsInGraph) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (Operation &nestedOp : block.getOperations()) {
          dfs(&nestedOp, opToTensorInfo, graph);
          dfsTraverse(&nestedOp, opToTensorInfo, graph, unsupportedOpsInGraph);
        }
      }
    }
  }

  Box<CppGraphConverter> create_egraph(std::unordered_map<int, Operation*> *unsupportedOpsInGraph, ModuleOp module) {
    auto graph = new_converter();
    std::unordered_map<Operation*, Box<TensorInfo>> opToTensorInfo;

    for (auto &op : module.getOps()) {
      dfs(&op, &opToTensorInfo, graph);
      dfsTraverse(&op, &opToTensorInfo, graph, unsupportedOpsInGraph);
    }

    return graph;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::unordered_map<int, Operation*> unsupportedOpsInGraph;

    create_egraph(&unsupportedOpsInGraph, module);

    // auto egraph_rep = dfs_insert_into_egraph(module);
    // egraph_rep.optimize();
    // extract
    // Walk through all operations and insert blackboxes for unsupported operations.
  }
};
} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createEqualitySaturationPass() {
  return std::make_unique<EqualitySaturationPass>();
}
} // end namespace enzyme
} // end namespace mlir
