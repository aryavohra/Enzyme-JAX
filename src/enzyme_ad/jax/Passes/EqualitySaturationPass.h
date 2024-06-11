#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

class OperationCreator {
public:
  OperationCreator(mlir::MLIRContext *context) : context(context), opBuilder(context) {}
  mlir::Operation *createAddOp(mlir::Operation *lhs, mlir::Operation *rhs);

private:
  mlir::MLIRContext* context;
  mlir::OpBuilder opBuilder;
};