#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/IR/BuiltinOps.h"
#include <fstream>
#include <iostream>

#define DEBUG_TYPE "enzyme"

using namespace mlir;


namespace {
class EqualitySaturationPass : public PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "equality-saturation-pass"; }
  StringRef getDescription() const override { return "Optimizes HLO graph using a Rust-based optimizer"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Simply walks through all operations, does nothing else.
    module.walk([](Operation *op) { 
      std::cout << "Operation name: " << op->getName().getStringRef().str() << "\n";
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