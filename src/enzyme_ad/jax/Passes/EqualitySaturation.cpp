#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/deps/include/ReactantExtra.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/executable.h"

#include "xla/python/pjrt_ifrt/xla_compiler.h"

#include "cxxbridge/deps/tensat/src/input.rs.h"
#include "rust/cxx.h"

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <memory>

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace rust::cxxbridge1;

class OperationMapInfo : public llvm::DenseMapInfo<Operation*> {
public:
  static unsigned getHashValue(const Operation* val) {
    return OperationEquivalence::computeHash(
      const_cast<Operation*>(val),
      // Operands, values and locations don't matter for runtime - we just
      // need the operation, attributes and types to be the same.
      OperationEquivalence::ignoreHashValue,
      OperationEquivalence::ignoreHashValue,
      OperationEquivalence::IgnoreLocations);
  }

  // Adapted from llvm-project/mlir/lib/Transforms/CSE.cpp
  static bool isEqual(const Operation* lhsC, const Operation* rhsC) {
    auto* lhs = const_cast<Operation*>(lhsC);
    auto* rhs = const_cast<Operation*>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        lhs, rhs,
        OperationEquivalence::ignoreValueEquivalence,
        nullptr,
        OperationEquivalence::IgnoreLocations);
  }
};

class OperationTimer {
public:
  /**
   * Measure cost of operation (execution time in microseconds) by running it many times and measuring the time taken.
   * TODO: Make cloning optional
   * TODO: Preserve context across runs so that we're not creating unnecessary contexts
  */
  static uint64_t getCost(Operation* op, unsigned warmup, unsigned repetitions) {
    if (!logsInitialized) {
      InitializeLogs();
      logsInitialized = true;
    }

    std::string opName = op->getName().getStringRef().str();

    // TODO: Have a whitelist instead?
    if (op->getDialect()->getNamespace() != "stablehlo" || opName == "stablehlo.constant" 
        || opName == "stablehlo.return" || opName == "stablehlo.compare")
      return 0;
    
    if (runtimeCache.contains(op)) {
      return runtimeCache[op];
    }

    DialectRegistry registry;
    InitializeRegistryAndPasses(wrap(&registry));

    MLIRContext context(registry);
    RegisterDialects(wrap(&context));

    ModuleOp wrapperModule = createModuleFromOperation(context, op);

    auto executable = prepareExecutable(wrapperModule, op);

    unsigned numResults = op->getNumResults();
    xla::PjRtBuffer* res[numResults];
    uint8_t futures = 0;

    auto t1 = std::chrono::high_resolution_clock::now();

    for (unsigned i = 0; i < warmup + repetitions; i++) {
      if (i == warmup) t1 = std::chrono::high_resolution_clock::now();
      XLAExecute(executable, 0, nullptr, nullptr, numResults, res, &futures, nullptr);
    }

    assert(!futures);

    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // Cleanup
    for (int i = 0; i < numResults; i++) {
      PjRtBufferFree(res[i]);
    }

    FreeClient(executable->client());
    ExecutableFree(executable);

    wrapperModule.erase();

    runtimeCache.try_emplace(op, duration);
    return duration;
  }

  // TODO: Look into using input ops, to avoid constant folding etc
  static Operation *getDummyOp(OpBuilder &builder, Type type) {
    // Zero-initialise inputs with same operand shape
    Attribute zeroAttr = builder.getZeroAttr(type);
    OperationState zeroState(builder.getUnknownLoc(), "stablehlo.constant");
    zeroState.addTypes(type);
    zeroState.addAttribute("value", zeroAttr);

    return builder.create(zeroState);
  }

private:
  static llvm::DenseMap<Operation*, uint64_t, OperationMapInfo> runtimeCache;

  inline static bool logsInitialized;

  /**
   * Create a clone of the operation in the new context recursively (i.e. going down to the regions).
   * Just using op->clone() will preserve context of the original operation, which poses a problem later
   * since stablehlo -> mhlo legalization pass will not match the new operation. 
   * 
   * Like the normal op->clone(), any operands that use values outside of the operations are remapped using 
   * the map that is provided (leaving them alone if no entry is present).
   * 
   * TODO: Surely there's a simpler way to do this?
  */
  static Operation *cloneOpInContext(OpBuilder &builder,
                                           Operation *op,
                                           IRMapping& mapping) {
    Location location = builder.getUnknownLoc();

    // Recursively clone regions
    llvm::SmallVector<std::unique_ptr<Region>> regions;

    for (auto& region : op->getRegions()) {
      auto newRegion = std::make_unique<Region>();

      for (auto& block : region.getBlocks()) {
        auto newBlock = new Block();

        // Map from old block arguments to new ones
        for (auto& arg : block.getArguments()) {
          mapping.map(arg, newBlock->addArgument(arg.getType(), location));
        }

        for (auto& nestedOp : block.getOperations()) {
          auto *newNestedOp = cloneOpInContext(builder, &nestedOp, mapping);  
          newBlock->push_back(newNestedOp);
          
          // Map result of old operation to that of new operation, so that operations after can use it
          for (int i = 0; i < nestedOp.getNumResults(); i++) {
            mapping.map(nestedOp.getResult(i), newNestedOp->getResult(i));
          }
        }
        newRegion->push_back(newBlock);
      }
      regions.push_back(std::move(newRegion));
    }

    OperationState opState(location, 
                            op->getName().getStringRef().str(),   // Use string to make a new name, rather than reusing the OperationName
                            op->getOperands(),
                            op->getResultTypes(),
                            op->getAttrs(),
                            {},
                            regions);

    auto *newOp = builder.create(opState);
    
    for (int i = 0; i < newOp->getNumOperands(); i++) {
      newOp->setOperand(i, mapping.lookupOrDefault(newOp->getOperand(i)));
    }

    return newOp;
  }

  static Operation *cloneOpInContext(OpBuilder &builder, Operation *op) {
    IRMapping mapping;
    return cloneOpInContext(builder, op, mapping);
  }

  /**
   * Wrap operation into a module, with dummy (constant zero) inputs as
   * its operands. Doesn't mutate op (instead it creates a copy).
   */
  static ModuleOp createModuleFromOperation(MLIRContext &context, Operation *op) {
    // Wrap operation into a module with dummy inputs
    OpBuilder builder(&context);
    Location location = builder.getUnknownLoc();
    ModuleOp wrapperModule = ModuleOp::create(location);

    auto block = wrapperModule.getBodyRegion().begin();

    auto *newOp = cloneOpInContext(builder, op);

    // Create a func.func to wrap newOp around
    FunctionType funcType = FunctionType::get(&context, {}, op->getResultTypes());
    func::FuncOp funcOp = builder.create<func::FuncOp>(location, "main", funcType);
    block->push_back(funcOp);

    Block *entryBlock = funcOp.addEntryBlock();

    llvm::SmallVector<Value> dummyInputs;
    auto operandTypes = op->getOperandTypes();

    for (auto type : operandTypes) {
      Operation *zeroOp = getDummyOp(builder, type);
      dummyInputs.push_back(zeroOp->getResult(0));
    }

    for (int i = 0; i < dummyInputs.size(); i++) {
      newOp->setOperand(i, dummyInputs[i]);
      entryBlock->push_back(dummyInputs[i].getDefiningOp());
    }

    entryBlock->push_back(newOp);
    
    auto returnOp = builder.create<func::ReturnOp>(location, newOp->getResults());
    entryBlock->push_back(returnOp);

    return std::move(wrapperModule);
  }

  /**
   * Wrap and compile operation into a PjRtLoadedExecutable, to be passed into XLAExecute.
  */
  static xla::PjRtLoadedExecutable* prepareExecutable(ModuleOp &wrapperModule, Operation *op) {
    if (failed(verify(wrapperModule))) {
      llvm::errs() << "Module verification error\n";
    }

    // TODO: GPU
    xla::PjRtClient *client = MakeCPUClient(0, 1, 1);

    xla::PjRtLoadedExecutable *executable = ClientCompile(client, wrap(wrapperModule));

    return executable;
  }
};

llvm::DenseMap<Operation*, uint64_t, OperationMapInfo> OperationTimer::runtimeCache;

// TODO: Avoid creating new MLIRContexts
// TODO: Avoid creating dummy inputs (we need them again for cost measurement, so duplicated)
// TODO: Lump binary ops together

uint64_t tensat::CostModel::getAddOpCost(rust::Slice<const int64_t> lhsDims,
                                              tensat::Type lhsType,
                                              rust::Slice<const int64_t> rhsDims,
                                              tensat::Type rhsType) const {
  DialectRegistry registry;
  InitializeRegistryAndPasses(wrap(&registry));

  MLIRContext context(registry);
  RegisterDialects(wrap(&context));

  OpBuilder builder(&context);

  auto lhs = OperationTimer::getDummyOp(builder, newTensorType(builder, lhsDims, lhsType));
  auto rhs = OperationTimer::getDummyOp(builder, newTensorType(builder, rhsDims, rhsType));

  auto addOp = builder.create<stablehlo::AddOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
  auto cost = OperationTimer::getCost(addOp, 100, 100);

  addOp.erase();

  return cost;  
}

uint64_t tensat::CostModel::getSubtractOpCost(rust::Slice<const int64_t> lhsDims,
                                              tensat::Type lhsType,
                                              rust::Slice<const int64_t> rhsDims,
                                              tensat::Type rhsType) const {
  DialectRegistry registry;
  InitializeRegistryAndPasses(wrap(&registry));

  MLIRContext context(registry);
  RegisterDialects(wrap(&context));

  OpBuilder builder(&context);

  auto lhs = OperationTimer::getDummyOp(builder, newTensorType(builder, lhsDims, lhsType));
  auto rhs = OperationTimer::getDummyOp(builder, newTensorType(builder, rhsDims, rhsType));

  auto subtractOp = builder.create<stablehlo::SubtractOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
  auto cost = OperationTimer::getCost(subtractOp, 100, 100);

  subtractOp.erase();

  return cost;  
}

uint64_t tensat::CostModel::getMulOpCost(rust::Slice<const int64_t> lhsDims,
                                              tensat::Type lhsType,
                                              rust::Slice<const int64_t> rhsDims,
                                              tensat::Type rhsType) const {
  DialectRegistry registry;
  InitializeRegistryAndPasses(wrap(&registry));

  MLIRContext context(registry);
  RegisterDialects(wrap(&context));

  OpBuilder builder(&context);

  auto lhs = OperationTimer::getDummyOp(builder, newTensorType(builder, lhsDims, lhsType));
  auto rhs = OperationTimer::getDummyOp(builder, newTensorType(builder, rhsDims, rhsType));

  auto mulOp = builder.create<stablehlo::MulOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
  auto cost = OperationTimer::getCost(mulOp, 100, 100);

  mulOp.erase();

  return cost;  
}

uint64_t tensat::CostModel::getDivOpCost(rust::Slice<const int64_t> lhsDims,
                                              tensat::Type lhsType,
                                              rust::Slice<const int64_t> rhsDims,
                                              tensat::Type rhsType) const {
  DialectRegistry registry;
  InitializeRegistryAndPasses(wrap(&registry));

  MLIRContext context(registry);
  RegisterDialects(wrap(&context));

  OpBuilder builder(&context);

  auto lhs = OperationTimer::getDummyOp(builder, newTensorType(builder, lhsDims, lhsType));
  auto rhs = OperationTimer::getDummyOp(builder, newTensorType(builder, rhsDims, rhsType));

  auto divOp = builder.create<stablehlo::DivOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
  auto cost = OperationTimer::getCost(divOp, 100, 100);

  divOp.erase();

  return cost;  
}

mlir::Type tensat::CostModel::newTensorType(OpBuilder& builder, rust::Slice<const int64_t> dims, tensat::Type type) const {
  auto dimsRef = llvm::ArrayRef(dims.data(), dims.size());
  auto mlirType = tensatTypeToMlirType(builder, type);
  return RankedTensorType::get(dimsRef, mlirType);
}

mlir::Type tensat::CostModel::tensatTypeToMlirType(OpBuilder& builder, tensat::Type type) const {
  switch (type) {
    case tensat::Type::i32:
      return builder.getI32Type();
    case tensat::Type::f32:
      return builder.getF32Type();
    default:
      return nullptr; // TODO: Probably not the best practice?
  }
}

std::unique_ptr<tensat::CostModel> tensat::newCostModel() {
  return std::make_unique<tensat::CostModel>();
}

namespace {
  class EqualitySaturationPass
    : public PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {
    public:
    StringRef getArgument() const override { return "equality-saturation-pass"; }
    StringRef getDescription() const override {
      return "Optimizes HLO graph using a Rust-based optimizer";
    }

    std::vector<int32_t> castArrayRefToInt32(llvm::ArrayRef<int64_t> shape) {
      std::vector<int32_t> dims;
      dims.reserve(shape.size());
      for (int64_t dim : shape) {
        dims.push_back(static_cast<int32_t>(dim));
      }
      return dims;
    }

    rust::Slice<const int32_t> castArrayRefToInt32Slice(llvm::ArrayRef<int64_t> input) {
      auto input_vec = castArrayRefToInt32(input);
      auto input_slice = rust::Slice<const int32_t>{
        input_vec.data(), static_cast<size_t>(input_vec.size())};
      return input_slice;
    }

    tensat::TensorInfo* handleEnodeOperand(
        Value operand,
        std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
        std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
        Box<tensat::CppGraphConverter> &graph) {
      if (auto defOp = operand.getDefiningOp()) {
        // Use existing TensorInfo if already processed
        return dfs(defOp, opToTensorInfo, blockArgToTensorInfo, graph);
      } else if (auto arg = operand.dyn_cast<BlockArgument>()) {
        // Handle BlockArguments which represent function parameters
        if (isa<TensorType>(operand.getType())) {
          int32_t block_arg_number = arg.getArgNumber();
          auto &tensorInfo = (*blockArgToTensorInfo)[block_arg_number];
          if (!tensorInfo) {
            auto shape = operand.getType().cast<TensorType>().getShape();
            auto dims = castArrayRefToInt32(shape);
            auto input_slice = rust::Slice<const int32_t>{
              dims.data(), static_cast<size_t>(dims.size())};
            tensorInfo = graph->new_input(block_arg_number, input_slice).into_raw();
            (*blockArgToTensorInfo)[block_arg_number] = tensorInfo;
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
      tensat::TensorInfo* handleOperation(
          Operation* op,
          CreateOpFunc createOpFunc,
          std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
          std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
          Box<tensat::CppGraphConverter> &graph,
          Args&&... args) {
        auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
        auto handleArgs = [&](auto&&... operands) {
          return std::make_tuple(handleEnodeOperand(operands, opToTensorInfo, blockArgToTensorInfo, graph)...);
        };

        // Apply handleArgs to unpack the tuple of operands into handleEnodeOperand calls
        auto operandInfos = std::apply(handleArgs, args_tuple);

        // Use std::apply to unpack operandInfos into the function call
        return std::apply([&](auto&&... unpacked) {
      return std::invoke(createOpFunc, *graph, *unpacked...).into_raw();
      }, operandInfos);
      }


      tensat::TensorInfo *dfs(Operation* op,
        std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
        std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
        Box<tensat::CppGraphConverter> &graph) {
      if (opToTensorInfo->find(op) != opToTensorInfo->end()) {
        return opToTensorInfo->at(op);
      }

      tensat::TensorInfo *tensorInfo = nullptr;

      if (isa<stablehlo::ConstantOp>(op)) {
        tensorInfo = graph->new_constant_op().into_raw();
      } else {
        auto handleEnodeOperandPartial = [&](Value operand) {
          return handleEnodeOperand(operand, opToTensorInfo, blockArgToTensorInfo, graph); 
        };
        auto handleOperationPartial = [&](auto&& createOpFunc, auto&&... operands) {
          return handleOperation(op, createOpFunc, opToTensorInfo, blockArgToTensorInfo, graph, std::forward<decltype(operands)>(operands)...);
        }; 

        if (isa<stablehlo::MulOp>(op)) {
          auto mul = cast<stablehlo::MulOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_mul_op, mul.getLhs(), mul.getRhs());
        } else if (isa<stablehlo::SubtractOp>(op)) {
          auto subtract = cast<stablehlo::SubtractOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_subtract_op, subtract.getLhs(), subtract.getRhs());
        } else if (isa<stablehlo::DivOp>(op)) {
          auto div = cast<stablehlo::DivOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_div_op, div.getLhs(), div.getRhs());
        } else if (isa<stablehlo::AddOp>(op)) {
          auto add = cast<stablehlo::AddOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_add_op, add.getLhs(), add.getRhs());
        } else if (isa<stablehlo::MinOp>(op)) {
          auto min = cast<stablehlo::MinOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_min_op, min.getLhs(), min.getRhs());
        } else if (isa<stablehlo::MaxOp>(op)) {
          auto max = cast<stablehlo::MaxOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_max_op, max.getLhs(), max.getRhs());
        } else if (isa<stablehlo::TanhOp>(op)) {
          auto tanh = cast<stablehlo::TanhOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_tanh_op, tanh.getOperand());
        } else if (isa<stablehlo::NegOp>(op)) {
          auto neg = cast<stablehlo::NegOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_neg_op, neg.getOperand());
        } else if (isa<stablehlo::ExpOp>(op)) {
          auto exp = cast<stablehlo::ExpOp>(op);
          tensorInfo = handleOperationPartial(&tensat::CppGraphConverter::new_exp_op, exp.getOperand());
        } else if (isa<stablehlo::TransposeOp>(op)) {
          auto transpose = cast<stablehlo::TransposeOp>(op);
          std::vector<int32_t> permutation = castArrayRefToInt32(transpose.getPermutation());
          auto permutation_slice = rust::Slice<const int32_t> {
            permutation.data(), static_cast<size_t>(permutation.size())};
            tensorInfo = graph->new_transpose_op(
              *handleEnodeOperandPartial(transpose.getOperand()),
              permutation_slice
            ).into_raw();
        } else if (isa<stablehlo::ReshapeOp>(op)) {
          auto reshape = cast<stablehlo::ReshapeOp>(op);
          if (auto output_tensor = reshape.getResult().getType().cast<TensorType>()) {
            auto shape = castArrayRefToInt32(output_tensor.getShape());
            auto output_shape_slice = rust::Slice<const int32_t> {
              shape.data(), static_cast<size_t>(shape.size())};
            tensorInfo = graph->new_reshape_op(
              *handleEnodeOperandPartial(reshape.getOperand()),
              output_shape_slice
            ).into_raw();
          } else {
            std::cout << "EqualitySaturationPass: result of stablehlo::ReshapeOp has non-tensor type" << std::endl;
          }
        } else if (isa<stablehlo::IotaOp>(op)) {
          auto iota = cast<stablehlo::IotaOp>(op);
          int32_t iota_dimension = iota.getIotaDimension();
          if (auto output_tensor = iota.getResult().getType().cast<TensorType>()) {
            auto shape = castArrayRefToInt32(output_tensor.getShape());
            auto output_shape_slice = rust::Slice<const int32_t>{
              shape.data(), static_cast<size_t>(shape.size())};
            tensorInfo = graph->new_iota_op(
              iota_dimension,
              output_shape_slice
            ).into_raw();
          } else {
            std::cout << "EqualitySaturationPass: result of stablehlo::IotaOp has non-tensor type" << std::endl;
          }
        } else if (isa<stablehlo::DotGeneralOp>(op)) {
          // we might need more guards here
          auto dot_general = cast<stablehlo::DotGeneralOp>(op);
          auto dot_dim_attrs = dot_general.getDotDimensionNumbersAttr();
          auto lhs_batch_dim = castArrayRefToInt32Slice(dot_dim_attrs.getLhsBatchingDimensions());
          auto rhs_batch_dim = castArrayRefToInt32Slice(dot_dim_attrs.getRhsBatchingDimensions());
          auto lhs_contracting_dim = castArrayRefToInt32Slice(dot_dim_attrs.getLhsContractingDimensions());
          auto rhs_contracting_dim = castArrayRefToInt32Slice(dot_dim_attrs.getRhsContractingDimensions());
          mlir::ArrayAttr precision = dot_general.getPrecisionConfig().value_or(mlir::ArrayAttr());
          std::vector<int> precision_configs;
          for (int i = 0; i < precision.size(); i++) {
            auto precisionAttr = precision[i].dyn_cast<mlir::stablehlo::PrecisionAttr>();
            if (!precisionAttr) continue;  // Skip if it's not a PrecisionAttr, although
                  // such attributes should not exist here
            mlir::stablehlo::Precision val = precisionAttr.getValue();
            switch (val) {
              case mlir::stablehlo::Precision::DEFAULT:
                precision_configs.push_back(0);
                break;
              case mlir::stablehlo::Precision::HIGH:
                precision_configs.push_back(1);
                break;
              case mlir::stablehlo::Precision::HIGHEST:
                precision_configs.push_back(2);
                break;
            }
          }
          auto precision_config_slice = rust::Slice<const int>{
            precision_configs.data(), static_cast<size_t>(precision_configs.size())};
          tensorInfo = graph->new_dot_general_op(
            *handleEnodeOperandPartial(dot_general.getLhs()),
            *handleEnodeOperandPartial(dot_general.getRhs()),
            lhs_batch_dim,
            rhs_batch_dim,
            lhs_contracting_dim,
            rhs_contracting_dim,
            precision_config_slice
          ).into_raw();
        }
      }

      if (tensorInfo != nullptr) {
        opToTensorInfo->insert({op, tensorInfo});
        return tensorInfo;
      }
      return tensorInfo;
    }

    Box<tensat::CppGraphConverter> createEgraph(
        std::unordered_map<int, Operation*> *unsupportedOpsInGraph,
        ModuleOp module) {

      auto graph = tensat::new_converter();
      std::unordered_map<Operation*, tensat::TensorInfo*> opToTensorInfo;
      std::unordered_map<int, tensat::TensorInfo*> blockArgToTensorInfo;

      module.walk([&](mlir::Operation *op) {
        std::cout << "ENTERING AT: " << op->getName().getStringRef().str() << "\n";
        auto cost = OperationTimer::getCost(op, 100, 100);
        llvm::outs() << "Cost: " << cost << "\n\n";
        dfs(op, &opToTensorInfo, &blockArgToTensorInfo, graph);
      });

      graph->print_rec_expr();
      return graph;
    }

    template <typename T>
    Operation* createUnaryOp(OpBuilder &builder, std::vector<Value>& opVals, tensat::Node &node) {
      auto location = builder.getUnknownLoc();
      return builder.create<T>(location, opVals[node.operands[0]]);
    }

    template <typename T>
    Operation* createBinaryOp(OpBuilder &builder, std::vector<Value>& opVals, tensat::Node &node) {
      auto location = builder.getUnknownLoc();
      return builder.create<T>(location, opVals[node.operands[0]], opVals[node.operands[1]]);
    }

    void reconstructStablehlo(ModuleOp *root, rust::vec<tensat::Node> &nodes) {
      auto context = root->getContext();
      OpBuilder builder(context);

      std::vector<Value> opVals;

      // Find funcOp to get the block.
      func::FuncOp funcOp;

      for (auto &op : root->getBody()->getOperations()) {
        if (isa<func::FuncOp>(op)) {
          funcOp = cast<func::FuncOp>(op);
          break;
        }
      }

      auto& region = funcOp.getRegion();
      auto& block = funcOp.getRegion().front();

      block.clear();

      auto location = builder.getUnknownLoc();

      for (auto& node : nodes) {
        Operation* newOp = nullptr;

        // Create the new operation based on the operands
        if (node.name == "NegOp") {
          newOp = createUnaryOp<stablehlo::NegOp>(builder, opVals, node);
        } else if (node.name == "TanhOp") {
          newOp = createUnaryOp<stablehlo::TanhOp>(builder, opVals, node);
        } else if (node.name == "ExpOp") {
          newOp = createUnaryOp<stablehlo::ExpOp>(builder, opVals, node);
        } else if (node.name == "AddOp") {
          newOp = createBinaryOp<stablehlo::AddOp>(builder, opVals, node);
        } else if (node.name == "SubtractOp") {
          newOp = createBinaryOp<stablehlo::SubtractOp>(builder, opVals, node);
        } else if (node.name == "MulOp") {
          newOp = createBinaryOp<stablehlo::MulOp>(builder, opVals, node);
        } else if (node.name == "DivOp") {
          newOp = createBinaryOp<stablehlo::DivOp>(builder, opVals, node);
        } else if (node.name == "MinOp") {
          newOp = createBinaryOp<stablehlo::MinOp>(builder, opVals, node);
        } else if (node.name == "MaxOp") {
          newOp = createBinaryOp<stablehlo::MaxOp>(builder, opVals, node);
        } else if (node.name == "Input") {
          int blockArgNumber = nodes[node.operands[1]].operands[0];
          opVals.push_back(block.getArgument(blockArgNumber));
          continue;
        } else {
          // TODO: implement other operations
          std::cout << node.name << "\n";
        }

        if (newOp) {
          block.push_back(newOp);
          opVals.push_back(newOp->getResult(0));
        } else {
          // This is bad practice, as we're pushing nullptr
          // to ops in case of Input, Num, or Var nodes. This
          // is unsafe, but maintains indexing. We could use
          // some llvm no-op, but that would not be much better.
          opVals.push_back(nullptr);
        }
      }

      assert(!block.empty());

      auto returnOp = builder.create<func::ReturnOp>(builder.getUnknownLoc(), block.back().getResults());
      block.push_back(returnOp);
    }

    void runOnOperation() override {
      ModuleOp module = getOperation();
      std::unordered_map<int, Operation*> unsupportedOpsInGraph;

      auto graph = createEgraph(&unsupportedOpsInGraph, module);
      
      auto optimized = graph->optimize();

      // TODO: Just for testing, remove later
      for (auto& node : optimized) {
        std::cout << node.name << ' ';
        for (auto& i : node.operands) {
          std::cout << i << ' ';
        }
        std::cout << std::endl;
      }

      std::cout << "reconstructing\n";
      reconstructStablehlo(&module, optimized);
      module.dump();
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
