#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "rust/cxx.h"

namespace tensat {
enum class Type : uint8_t;
enum class Ops : uint8_t;
struct Shape;

/**
 * Functions exposed to Rust (Tensat) for getting the cost of new operations.
 */
class CostModel {
public:
  // TODO: Our operands should be tuples of (shape, type).
  // SHOULD TensorData have a type field too?
  uint64_t getBinOpCost(Ops op, rust::Slice<const int64_t> lhsDims,
                        Type lhsType, rust::Slice<const int64_t> rhsDims,
                        tensat::Type rhsType) const;

  static mlir::Type newTensorType(mlir::OpBuilder &builder,
                           rust::Slice<const int64_t> dims,
                           tensat::Type type);
  static mlir::Type tensatTypeToMlirType(mlir::OpBuilder &builder,
                                  tensat::Type type);
};

class ShapeInference {
public:

  rust::Vec<Shape> get_shape(
    Ops op,
    rust::Slice<const rust::Slice<const int64_t>> operand_dims,
    rust::Slice<const Type> operand_types,
    rust::Slice<const rust::Slice<const int64_t>> other_vector_args,
    rust::Slice<const int64_t> int_args) const;
};

std::unique_ptr<tensat::CostModel> newCostModel();
std::unique_ptr<tensat::ShapeInference> newShapeInference();
} // namespace tensat
