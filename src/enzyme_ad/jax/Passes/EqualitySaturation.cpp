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
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "src/enzyme_ad/jax/deps/include/ReactantExtra.h"

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/executable.h"

#include "xla/python/pjrt_ifrt/xla_compiler.h"

#include <fstream>
#include <iostream>

#define DEBUG_TYPE "enzyme"


namespace {
class EqualitySaturationPass : public mlir::PassWrapper<EqualitySaturationPass, mlir::OperationPass<mlir::ModuleOp>> {

  mlir::StringRef getArgument() const override { return "equality-saturation-pass"; }
  mlir::StringRef getDescription() const override { return "Optimizes HLO graph using a Rust-based optimizer"; }

  /**
   * Wrap operation into a module, with dummy (constant zero) inputs as its operands.
   * Doesn't mutate op (instead it creates a copy).
  */
  static mlir::ModuleOp createModuleFromOperation(mlir::MLIRContext& context, mlir::Operation* op) {
    // Wrap operation into a module with dummy inputs
    mlir::OpBuilder builder(&context);
    mlir::Location location = builder.getUnknownLoc();
    mlir::ModuleOp wrapperModule = mlir::ModuleOp::create(location);

    auto block = wrapperModule.getBodyRegion().begin();

    mlir::Operation::CloneOptions cloneOptions;
    cloneOptions.cloneOperands();

    mlir::Operation *newOp = op->clone(cloneOptions);

    // Create a func.func to wrap newOp arounds

    mlir::FunctionType funcType = mlir::FunctionType::get(&context, {}, newOp->getResultTypes());

    mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(location, "main", funcType);
    block->push_back(funcOp);

    mlir::Block *entryBlock = funcOp.addEntryBlock();

    auto operandTypes = newOp->getOperandTypes();

    llvm::SmallVector<mlir::Value> dummyInputs;

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

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    // Simply walks through all operations, does nothing else.

    modOp.walk([](mlir::Operation *op) {
      std::string opName = op->getName().getStringRef().str();
      
      std::cout << "Operation name: " << opName << "\n";

      mlir::DialectRegistry registry;
      InitializeRegistryAndPasses(wrap(&registry));

      mlir::MLIRContext context(registry);
      RegisterDialects(wrap(&context));

      // Wrap operation into a module with dummy inputs
      mlir::OpBuilder builder(&context);
      mlir::Location location = builder.getUnknownLoc();
      mlir::ModuleOp wrapperModule = EqualitySaturationPass::createModuleFromOperation(context, op);

      std::cout << "Verifying module\n";

      if (mlir::failed(mlir::verify(wrapperModule))) {
        llvm::errs() << "Module verification error\n";
      }

      std::cout << "Printing module" << std::endl;
      // Print the module
      wrapperModule.print(llvm::outs());
      llvm::outs() << "\n";
      /*
      // TODO: GPU
      xla::PjRtClient* client = MakeCPUClient(0, 1, 1);

      std::cout << "Compiling\n";
      xla::PjRtLoadedExecutable* executable = ClientCompile(client, wrap(wrapperModule));
      std::cout << "Compiled executable\n";

      std::cout << "Executing\n";
      XLAExecute(executable, 0, nullptr, nullptr, 0, nullptr, nullptr, nullptr);
      std::cout << "Executed\n";
      */
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