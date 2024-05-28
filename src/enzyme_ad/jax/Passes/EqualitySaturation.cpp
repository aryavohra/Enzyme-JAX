#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"
#include "src/enzyme_ad/jax/deps/include/ReactantExtra.h"

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/executable.h"

#include "xla/python/pjrt_ifrt/xla_compiler.h"

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

#define DEBUG_TYPE "enzyme"

// bool logsInitialized = false;

class OperationTimer {
public:
  /**
   * Measure cost of operation (execution time in microseconds) by running it many times and measuring the time taken.
  */
  static uint64_t getCost(mlir::Operation* op, unsigned warmup, unsigned repetitions) {
    if (!logsInitialized) {
      InitializeLogs();
      logsInitialized = true;
    }

    auto executable = prepareExecutable(op);

    unsigned numResults = op->getNumResults();
    xla::PjRtBuffer* res[numResults];
    uint8_t futures = 0;

    auto t1 = std::chrono::high_resolution_clock::now();

    for (unsigned i = 0; i < warmup + repetitions; i++) {
      if (i == warmup) t1 = std::chrono::high_resolution_clock::now();
      XLAExecute(executable, 0, nullptr, nullptr, numResults, res, &futures, nullptr);
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    auto dur = t2 - t1;
    return std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
  }

private:
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
  static mlir::Operation *cloneOpInContext(mlir::OpBuilder &builder,
                                           mlir::Operation *op,
                                           mlir::IRMapping& mapping) {
    mlir::Location location = builder.getUnknownLoc();

    // Recursively clone regions
    llvm::SmallVector<std::unique_ptr<mlir::Region>> regions;

    for (auto& region : op->getRegions()) {
      auto newRegion = std::make_unique<mlir::Region>();

      for (auto& block : region.getBlocks()) {
        auto newBlock = new mlir::Block();

        // Map from old block arguments to new ones
        for (auto arg : block.getArguments()) {
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

    mlir::OperationState opState(location, 
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

  static mlir::Operation *cloneOpInContext(mlir::OpBuilder &builder, mlir::Operation *op) {
    mlir::IRMapping mapping;
    return cloneOpInContext(builder, op, mapping);
  }

  /**
   * Wrap operation into a module, with dummy (constant zero) inputs as
   * its operands. Doesn't mutate op (instead it creates a copy).
   */
  static mlir::ModuleOp createModuleFromOperation(mlir::MLIRContext &context, mlir::Operation *op) {
    // Wrap operation into a module with dummy inputs
    mlir::OpBuilder builder(&context);
    mlir::Location location = builder.getUnknownLoc();
    mlir::ModuleOp wrapperModule = mlir::ModuleOp::create(location);

    auto block = wrapperModule.getBodyRegion().begin();

    auto *newOp = cloneOpInContext(builder, op);

    // Create a func.func to wrap newOp around
    mlir::FunctionType funcType = mlir::FunctionType::get(&context, {}, op->getResultTypes());
    mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(location, "main", funcType);
    block->push_back(funcOp);

    mlir::Block *entryBlock = funcOp.addEntryBlock();

    llvm::SmallVector<mlir::Value> dummyInputs;
    auto operandTypes = op->getOperandTypes();

    for (auto type : operandTypes) {
      // Zero-initialise inputs with same operand shape
      mlir::Attribute zeroAttr = builder.getZeroAttr(type);
      mlir::OperationState zeroState(location, "stablehlo.constant");
      zeroState.addTypes(type);
      zeroState.addAttribute("value", zeroAttr);

      mlir::Operation *zeroOp = builder.create(zeroState);
      dummyInputs.push_back(zeroOp->getResult(0));
    }

    for (int i = 0; i < dummyInputs.size(); i++) {
      newOp->setOperand(i, dummyInputs[i]);
      entryBlock->push_back(dummyInputs[i].getDefiningOp());
    }

    entryBlock->push_back(newOp);
    
    auto returnOp = builder.create<mlir::func::ReturnOp>(location, newOp->getResults());
    entryBlock->push_back(returnOp);

    return wrapperModule;
  }

  /**
   * Wrap and compile operation into a PjRtLoadedExecutable, to be passed into XLAExecute.
  */
  static xla::PjRtLoadedExecutable* prepareExecutable(mlir::Operation *op) {
    mlir::DialectRegistry registry;
    InitializeRegistryAndPasses(wrap(&registry));

    mlir::MLIRContext context(registry);
    RegisterDialects(wrap(&context));

    mlir::ModuleOp wrapperModule = createModuleFromOperation(context, op);
    // llvm::outs() << wrapperModule << '\n';

    if (mlir::failed(mlir::verify(wrapperModule))) {
      llvm::errs() << "Module verification error\n";
    }

    // TODO: GPU
    xla::PjRtClient *client = MakeCPUClient(0, 1, 1);

    xla::PjRtLoadedExecutable *executable = ClientCompile(client, wrap(wrapperModule));

    return executable;
  }
};

namespace {
class EqualitySaturationPass : public mlir::PassWrapper<EqualitySaturationPass, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::StringRef getArgument() const override { return "equality-saturation-pass"; }
  mlir::StringRef getDescription() const override { return "Optimizes HLO graph using a Rust-based optimizer"; }

  void runOnOperation() override {    
    mlir::ModuleOp modOp = getOperation();

    modOp.walk([](mlir::Operation *op) {
      std::string opName = op->getName().getStringRef().str();
      llvm::outs() << "Operation name: " << opName << "\n";
      // TODO: have a whitelist in the cost function (returning 0 for everything else) instead?
      if (op->getDialect()->getNamespace() != "stablehlo" || opName == "stablehlo.constant" || opName == "stablehlo.return" || opName == "stablehlo.compare") return;

      auto cost = OperationTimer::getCost(op, 100, 100);
      llvm::outs() << "Cost: " << cost << "\n\n";
    });
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