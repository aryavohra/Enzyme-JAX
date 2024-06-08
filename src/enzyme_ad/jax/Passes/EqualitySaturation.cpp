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

  std::vector<int32_t> castArrayRefToInt32(llvm::ArrayRef<int64_t> shape) {
    std::vector<int32_t> dims;
    dims.reserve(shape.size());
    for (int64_t dim : shape) {
      dims.push_back(static_cast<int32_t>(dim));
    }
    return dims;
  }
    // we might want to just store pointers to TensorInfos instead of boxes.
  TensorInfo* dfs(
      Operation *op,
      std::unordered_map<Operation*, TensorInfo*> *opToTensorInfo,
      Box<CppGraphConverter> &graph
      ) {
    if (opToTensorInfo->find(op) != opToTensorInfo->end()) {
        return opToTensorInfo->at(op);
    }
    TensorInfo* tensorInfo = nullptr;
    if (isa<stablehlo::ConstantOp>(op)) {
	tensorInfo = graph->new_constant_op(measureCost(op)).into_raw();
    }
    else if (isa<stablehlo::MulOp>(op)) {
	auto mulOp = dyn_cast<stablehlo::MulOp>(op);
	auto lhs = mulOp.getLhs();
	auto rhs = mulOp.getRhs();
	TensorInfo* lhsTensorInfo = nullptr;
	TensorInfo* rhsTensorInfo = nullptr;

	if (lhs.getDefiningOp() == nullptr) {
	  // you are a blockargument
	  // if you are a tensortype, get shape
	  if (isa<TensorType>(lhs.getType())) {
	    auto shape = lhs.getType().getShape();
            auto dims = castArrayRefToInt32(shape);
            auto input_slice = rust::Slice<const int32_t>{dims.data(), static_cast<size_t>(dims.size())};
	    lhsTensorInfo = graph->new_input(input_slice).into_raw();
          } else {
	    lhs.getType().dump();
	  }
	} else {
	  lhsTensorInfo = dfs(lhs.getDefiningOp(), opToTensorInfo, graph);
	}
	if (rhs.getDefiningOp() == nullptr) {
	  // you are a blockargument
          // not handling booleans and complex numbers
	  if (isa<TensorType>(rhs.getType())) {
	    auto shape = rhs.getType().getShape();
            auto dims = castArrayRefToInt32(shape);
            auto input_slice = rust::Slice<const int32_t>{dims.data(), static_cast<size_t>(dims.size())};
            rhsTensorInfo = graph->new_input(input_slice).into_raw();
	  } else {
	    rhs.getType().dump();
	  }
	} else {
	  rhsTensorInfo = dfs(rhs.getDefiningOp(), opToTensorInfo, graph);
	}
	tensorInfo = graph->new_mul_op(*lhsTensorInfo, *rhsTensorInfo, measureCost(op)).into_raw();
    }
    if (tensorInfo != nullptr) {
	opToTensorInfo->insert({op, tensorInfo});
	return tensorInfo;
    }
    return tensorInfo;
 //    else {
	// std::cout << "default!" << "\n"; 
 //    }
  }
  
  Box<CppGraphConverter> create_egraph(std::unordered_map<int, Operation*> *unsupportedOpsInGraph, ModuleOp module) {
    auto graph = new_converter();
    std::unordered_map<Operation*, TensorInfo*> opToTensorInfo;

    // for (auto &op : module.getOps()) {
    //   dfs(&op, &opToTensorInfo, graph);
    // }
    module.walk([&](mlir::Operation *op) {
      std::cout << "ENTERING AT: " << op->getName().getStringRef().str() << "\n";
      dfs(op, &opToTensorInfo, graph);
    });
    graph->print_rec_expr();

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
