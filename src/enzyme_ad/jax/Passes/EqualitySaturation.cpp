#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>

#define DEBUG_TYPE "enzyme"

using namespace mlir;


namespace {
class EqualitySaturationPass : public PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "equality-saturation-pass"; }
  StringRef getDescription() const override { return "Optimizes HLO graph using a Rust-based optimizer"; }

  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    // Simply walks through all operations, does nothing else.

    modOp.walk([](Operation *op) {
      MLIRContext context;

      // Wrap operation into a module with dummy inputs

      OpBuilder builder(&context);
      Location location = builder.getUnknownLoc();
      ModuleOp wrapperModule = ModuleOp::create(location);

      auto block = wrapperModule.getBodyRegion().begin();
      
      Operation::CloneOptions cloneOptions;
      cloneOptions.cloneOperands();

      Operation* newOp = op->clone(cloneOptions);

      auto operandTypes = newOp->getOperandTypes();
      std::cout << "Operation name: " << newOp->getName().getStringRef().str() << "\n";

      llvm::SmallVector<Value> dummyInputs;

      for (auto type : operandTypes) {
        // Zero-initialise inputs with same operand shape
        Attribute zeroAttr = builder.getZeroAttr(type);
        OperationState zeroState(location, "arith.constant");
        zeroState.addTypes(type);
        zeroState.addAttribute("value", zeroAttr);

        Operation* zeroOp = builder.create(zeroState);
        dummyInputs.push_back(zeroOp->getResult(0));
      }

      for (int i = 0; i < dummyInputs.size(); i++) {
        newOp->setOperand(i, dummyInputs[i]);
        block->push_back(dummyInputs[i].getDefiningOp());
      }
      block->push_back(newOp);
      
      if (mlir::failed(mlir::verify(wrapperModule))) {
        llvm::errs() << "Module verification error\n";
      }

      // Print the module
      wrapperModule.print(llvm::outs());
      llvm::outs() << "\n";
      
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