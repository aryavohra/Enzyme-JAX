#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Passes/target/cxxbridge/rust/cxx.h"
#include "src/enzyme_ad/jax/Passes/target/cxxbridge/tensat/src/input.rs.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace rust::cxxbridge1;

namespace {
class EqualitySaturationPass
    : public PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const override { return "equality-saturation-pass"; }
  StringRef getDescription() const override {
    return "Optimizes HLO graph using a Rust-based optimizer";
  }

  int measureCost(Operation *op) { return 0; }

  std::vector<int32_t> castArrayRefToInt32(llvm::ArrayRef<int64_t> shape) {
    std::vector<int32_t> dims;
    dims.reserve(shape.size());
    for (int64_t dim : shape) {
      dims.push_back(static_cast<int32_t>(dim));
    }
    return dims;
  }

  TensorInfo* handleOperand(
      Value operand,
      std::unordered_map<Operation*, TensorInfo*> *opToTensorInfo,
      std::unordered_map<int, TensorInfo*> *blockArgToTensorInfo,
      Box<CppGraphConverter> &graph) {
    if (auto defOp = operand.getDefiningOp()) {
      // Use existing TensorInfo if already processed
      return dfs(defOp, opToTensorInfo, blockArgToTensorInfo, graph);
    } else if (auto arg = operand.dyn_cast<BlockArgument>()) {
      // Handle BlockArguments which represent function parameters
      if (isa<TensorType>(operand.getType())) {
        auto &tensorInfo = (*blockArgToTensorInfo)[arg.getArgNumber()];
        if (!tensorInfo) {
          auto shape = operand.getType().cast<TensorType>().getShape();
          auto dims = castArrayRefToInt32(shape);
          auto input_slice = rust::Slice<const int32_t>{
              dims.data(), static_cast<size_t>(dims.size())};
          tensorInfo = graph->new_input(input_slice).into_raw();
          (*blockArgToTensorInfo)[arg.getArgNumber()] = tensorInfo;
        }
        return tensorInfo;
      } else {
        std::cout
            << "EqualitySaturationPass does not support this argument type!"
            << "\n";
        operand.getType().dump();
        return nullptr;
      }
    }
    std::cout
        << "EqualitySaturationPass: encountered operand that is neither the result of an Op nor a BlockArgument."
        << "\n";
    return nullptr;
  }

  template <typename CreateOpFunc, typename... Args>
  TensorInfo* handleOperation(
        Operation* op,
        CreateOpFunc createOpFunc,
        std::unordered_map<Operation*, TensorInfo*> *opToTensorInfo,
        std::unordered_map<int, TensorInfo*> *blockArgToTensorInfo,
        Box<CppGraphConverter> &graph,
        Args&&... args) {
    auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
    auto handleArgs = [&](auto&&... operands) {
        return std::make_tuple(handleOperand(operands, opToTensorInfo, blockArgToTensorInfo, graph)...);
    };

    // Apply handleArgs to unpack the tuple of operands into handleOperand calls
    auto operandInfos = std::apply(handleArgs, args_tuple);

    // Use std::apply to unpack operandInfos into the function call
    return std::apply([&](auto&&... unpacked) {
        return std::invoke(createOpFunc, *graph, *unpacked..., measureCost(op)).into_raw();
    }, operandInfos);
  }


  TensorInfo *dfs(Operation* op,
                  std::unordered_map<Operation*, TensorInfo*> *opToTensorInfo,
                  std::unordered_map<int, TensorInfo*> *blockArgToTensorInfo,
                  Box<CppGraphConverter> &graph) {
    if (opToTensorInfo->find(op) != opToTensorInfo->end()) {
      return opToTensorInfo->at(op);
    }
    TensorInfo *tensorInfo = nullptr;

    if (isa<stablehlo::ConstantOp>(op)) {
      tensorInfo = graph->new_constant_op(measureCost(op)).into_raw();
    } else {
      auto handleOperandPartial = [&](Value operand) {
        return handleOperand(operand, opToTensorInfo, blockArgToTensorInfo, graph); 
      };
      auto handleOperationPartial = [&](auto&& createOpFunc, auto&&... operands) {
        return handleOperation(op, createOpFunc, opToTensorInfo, blockArgToTensorInfo, graph, std::forward<decltype(operands)>(operands)...);
      }; 

      if (isa<stablehlo::MulOp>(op)) {
        auto binaryOp = cast<stablehlo::MulOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_mul_op, binaryOp.getLhs(), binaryOp.getRhs());
      } else if (isa<stablehlo::SubtractOp>(op)) {
        auto binaryOp = cast<stablehlo::SubtractOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_subtract_op, binaryOp.getLhs(), binaryOp.getRhs());
      } else if (isa<stablehlo::DivOp>(op)) {
        auto binaryOp = cast<stablehlo::DivOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_div_op, binaryOp.getLhs(), binaryOp.getRhs());
      } else if (isa<stablehlo::AddOp>(op)) {
        auto binaryOp = cast<stablehlo::AddOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_add_op, binaryOp.getLhs(), binaryOp.getRhs());
      } else if (isa<stablehlo::MinOp>(op)) {
        auto binaryOp = cast<stablehlo::MinOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_min_op, binaryOp.getLhs(), binaryOp.getRhs());
      } else if (isa<stablehlo::MaxOp>(op)) {
        auto binaryOp = cast<stablehlo::MaxOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_max_op, binaryOp.getLhs(), binaryOp.getRhs());
      } else if (isa<stablehlo::TanhOp>(op)) {
        auto unaryOp = cast<stablehlo::TanhOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_tanh_op, unaryOp.getOperand());
      } else if (isa<stablehlo::NegOp>(op)) {
        auto unaryOp = cast<stablehlo::NegOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_neg_op, unaryOp.getOperand());
      } else if (isa<stablehlo::ExpOp>(op)) {
        auto unaryOp = cast<stablehlo::ExpOp>(op);
        tensorInfo = handleOperationPartial(&CppGraphConverter::new_exp_op, unaryOp.getOperand());
      } else if (isa<stablehlo::TransposeOp>(op)) {
        auto binaryOp = cast<stablehlo::TransposeOp>(op);
        std::vector<int32_t> permutation = castArrayRefToInt32(binaryOp.getPermutation());
        auto permutation_slice = rust::Slice<const int32_t>{
              permutation.data(), static_cast<size_t>(permutation.size())};
        tensorInfo = graph->new_transpose_op(
          *handleOperandPartial(binaryOp.getOperand()),
          permutation_slice,
          measureCost(op)
        ).into_raw();
      }
    }

    if (tensorInfo != nullptr) {
      opToTensorInfo->insert({op, tensorInfo});
      return tensorInfo;
    }
    return tensorInfo;
  }

  Box<CppGraphConverter> create_egraph(
      std::unordered_map<int, Operation*> *unsupportedOpsInGraph,
      ModuleOp module) {
    auto graph = new_converter();
    std::unordered_map<Operation*, TensorInfo*> opToTensorInfo;
    std::unordered_map<int, TensorInfo*> blockArgToTensorInfo;
    module.walk([&](mlir::Operation *op) {
      std::cout << "ENTERING AT: " << op->getName().getStringRef().str()
                << "\n";
      dfs(op, &opToTensorInfo, &blockArgToTensorInfo, graph);
    });
    graph->print_rec_expr();

    return graph;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::unordered_map<int, Operation*> unsupportedOpsInGraph;

    create_egraph(&unsupportedOpsInGraph, module);
  }
};
}  // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createEqualitySaturationPass() {
  return std::make_unique<EqualitySaturationPass>();
}
}  // end namespace enzyme
}  // end namespace mlir

