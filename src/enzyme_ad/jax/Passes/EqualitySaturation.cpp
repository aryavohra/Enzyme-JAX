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
  MLIRContext context;
  OpBuilder builder(&context);

  auto lhs = OperationTimer::getDummyOp(builder, newTensorType(builder, lhsDims, lhsType));
  auto rhs = OperationTimer::getDummyOp(builder, newTensorType(builder, rhsDims, rhsType));

  auto addOp = builder.create<stablehlo::AddOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
  auto cost = OperationTimer::getCost(addOp, 100, 100);

  addOp.erase();

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
    StringRef getArgument() const override {
      return "equality-saturation-pass";
    }
    StringRef getDescription() const override {
      return "Optimizes HLO graph using a Rust-based optimizer";
    }

    void runOnOperation() override {
      ModuleOp modOp = getOperation();

      modOp.walk([](Operation *op) {
        llvm::outs() << "Operation name: " << op->getName() << "\n";

        auto cost = OperationTimer::getCost(op, 100, 100);
        llvm::outs() << "Cost: " << cost << "\n\n";
      });

      modOp.dump();
      llvm::outs() << "\n";
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
