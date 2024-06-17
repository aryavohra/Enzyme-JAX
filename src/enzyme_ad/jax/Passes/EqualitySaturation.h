#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "rust/cxx.h"

namespace tensat {
        enum class Type: uint8_t;

        /**
         * Functions exposed to Rust (Tensat) for getting the cost of new operations.
         */
        class CostModel {
                public:
                        uint64_t getAddOpCost(rust::Slice<const int64_t> lhsDims,
                                        Type lhsType,
                                        rust::Slice<const int64_t> rhsDims,
                                        tensat::Type rhsType) const;
                        uint64_t getMulOpCost(rust::Slice<const int64_t> lhsDims,
                                        Type lhsType,
                                        rust::Slice<const int64_t> rhsDims,
                                        tensat::Type rhsType) const;
                        uint64_t getSubtractOpCost(rust::Slice<const int64_t> lhsDims,
                                        Type lhsType,
                                        rust::Slice<const int64_t> rhsDims,
                                        tensat::Type rhsType) const;
                        uint64_t getDivOpCost(rust::Slice<const int64_t> lhsDims,
                                        Type lhsType,
                                        rust::Slice<const int64_t> rhsDims,
                                        tensat::Type rhsType) const;

                private:
                        mlir::Type newTensorType(mlir::OpBuilder &builder,
                                        rust::Slice<const int64_t> dims,
                                        tensat::Type type) const;
                        mlir::Type tensatTypeToMlirType(mlir::OpBuilder &builder,
                                        tensat::Type type) const;
        };

        std::unique_ptr<tensat::CostModel> newCostModel();
}

