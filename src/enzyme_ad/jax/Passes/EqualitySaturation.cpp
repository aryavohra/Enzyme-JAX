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
#include <sstream>

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
      std::cout << "ENTERED CACHE" << "\n";
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

    // std::cout << op->getName().getStringRef().str() << "\n";
    // runtimeCache.try_emplace(op, duration);
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

  static Operation *cloneOpInContext(OpBuilder &builder, Operation *op) {
    IRMapping mapping;
    return cloneOpInContext(builder, op, mapping);
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

/**
* Create a new mlir::Type based on the element type of an existing mlir::Type and the provided shape.
*/
mlir::Type deriveOutputType(mlir::Value &input, llvm::ArrayRef<int64_t> shape) {
  auto inputType = input.getType();
  assert(isa<TensorType>(inputType));
  auto elementType = inputType.cast<TensorType>().getElementType();
  auto newType = RankedTensorType::get(shape, elementType);
  return newType;
}

std::vector<int64_t> rust_slice_to_cpp_vector(rust::Slice<const int64_t> input_slice) {
    std::vector<int64_t> result;
    for (const auto& value : input_slice) {
      result.push_back(value);
    }
    return result;
}

// TODO: Avoid creating new MLIRContexts
// TODO: Avoid creating dummy inputs (we need them again for cost measurement, so duplicated)
// TODO: Lump binary ops together
// TOD: Single function for all ops
uint64_t tensat::CostModel::get_cost(
    Ops op,
    rust::Slice<const rust::Slice<const int64_t>> operand_dims,
    rust::Slice<const tensat::Type> operand_types,
    rust::Slice<const rust::Slice<const int64_t>> other_vector_args,
    rust::Slice<const int64_t> int_args) const {
  DialectRegistry registry;
  InitializeRegistryAndPasses(wrap(&registry));

  MLIRContext context(registry);
  RegisterDialects(wrap(&context));

  OpBuilder builder(&context);

  SmallVector<Value> operands;
  for (const auto& [dim_slice, type] : llvm::zip(operand_dims, operand_types)) {
    auto tensor_type = tensat::CostModel::newTensorType(builder, dim_slice, type);
    operands.push_back(OperationTimer::getDummyOp(builder, tensor_type)->getResult(0));
  }

  std::vector<std::vector<int64_t>> other_vecs;
  for (const auto& vec : other_vector_args)
    other_vecs.push_back(rust_slice_to_cpp_vector(vec));

  Operation* mlirOp = nullptr;
  std::vector<int64_t> permutation_vec, shape_vec, lhs_batch_dim, rhs_batch_dim, lhs_contract_dim, rhs_contract_dim, precision_config, shape;
  std::vector<int64_t> start_indices, limit_indices, strides;
  std::vector<Attribute> precisionVec;

  switch (op) {
    case tensat::Ops::AddOp:
      mlirOp = builder.create<stablehlo::AddOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::MulOp:
      mlirOp = builder.create<stablehlo::MulOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::DivOp:
      mlirOp = builder.create<stablehlo::DivOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::SubtractOp:
      mlirOp = builder.create<stablehlo::SubtractOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::NegOp:
      mlirOp = builder.create<stablehlo::NegOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::TanhOp:
      mlirOp = builder.create<stablehlo::TanhOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::ExpOp:
      mlirOp = builder.create<stablehlo::ExpOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::MinOp:
      mlirOp = builder.create<stablehlo::MinOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::MaxOp:
      mlirOp = builder.create<stablehlo::MaxOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::TransposeOp:
      mlirOp = builder.create<stablehlo::TransposeOp>(builder.getUnknownLoc(), operands[0], other_vecs[0]);
      break;
    case tensat::Ops::ReshapeOp:
      mlirOp = builder.create<stablehlo::ReshapeOp>(builder.getUnknownLoc(), deriveOutputType(operands[0], other_vecs[0]), operands[0]);
      break;
    case tensat::Ops::DotGeneralOp:
      lhs_batch_dim = other_vecs[0];
      rhs_batch_dim = other_vecs[1];
      lhs_contract_dim = other_vecs[2];
      rhs_contract_dim = other_vecs[3];
      precision_config = other_vecs[4];
      shape = other_vecs[5];

      for (auto& precision : precision_config) {
        switch (precision) {
          case 0:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(&context, stablehlo::Precision::DEFAULT)); break;
          case 1:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(&context, stablehlo::Precision::HIGH)); break;
          case 2:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(&context, stablehlo::Precision::HIGHEST)); break;
        }
      }

      mlirOp = builder.create<stablehlo::DotGeneralOp>(
        builder.getUnknownLoc(),
        deriveOutputType(operands[1], shape),
        operands[0],
        operands[1],
        stablehlo::DotDimensionNumbersAttr::get(&context, lhs_batch_dim, rhs_batch_dim, lhs_contract_dim, rhs_contract_dim),
        mlir::ArrayAttr::get(&context, llvm::ArrayRef(precisionVec))
      );
      break;
    case tensat::Ops::SliceOp:
      start_indices = other_vecs[0];
      limit_indices = other_vecs[1];
      strides = other_vecs[2];
      mlirOp = builder.create<stablehlo::SliceOp>(
        builder.getUnknownLoc(),
	operands[0], 
	start_indices,
	limit_indices,
	strides
      );
      break;
    default:
      std::cout << "EGRAPH INVALID, UNSUPPORTED OP SHAPE REQUESTED" << "\n";
      assert(false);
      return {};
  }

  if (mlirOp) {
    auto cost = OperationTimer::getCost(mlirOp, 100, 100);
    mlirOp->erase();
    return cost;
  }
  return 100000;  
}

mlir::Type tensat::CostModel::newTensorType(OpBuilder& builder, rust::Slice<const int64_t> dims, tensat::Type type) {
  auto dimsRef = llvm::ArrayRef(dims.data(), dims.size());
  auto mlirType = tensatTypeToMlirType(builder, type);
  return RankedTensorType::get(dimsRef, mlirType);
}

mlir::Type tensat::CostModel::tensatTypeToMlirType(OpBuilder& builder, tensat::Type type) {
  switch (type) {
    case tensat::Type::i32:
      return builder.getI32Type();
    case tensat::Type::f32:
      return builder.getF32Type();
    default:
      assert(false);
  }
}

std::unique_ptr<tensat::CostModel> tensat::newCostModel() {
  return std::make_unique<tensat::CostModel>();
}

// SHAPE INFERENCE 
std::vector<int32_t> castArrayRefToInt32(llvm::ArrayRef<int64_t> shape) {
  std::vector<int32_t> dims;
  dims.reserve(shape.size());
  for (int64_t dim : shape) {
    dims.push_back(static_cast<int32_t>(dim));
  }
  return dims;
}

rust::Vec<tensat::Shape> tensat::ShapeInference::get_shape(
    Ops op,
    rust::Slice<const rust::Slice<const int64_t>> operand_dims,
    rust::Slice<const tensat::Type> operand_types,
    rust::Slice<const rust::Slice<const int64_t>> other_vector_args,
    rust::Slice<const int64_t> int_args) const {
  DialectRegistry registry;
  InitializeRegistryAndPasses(wrap(&registry));

  MLIRContext context(registry);
  RegisterDialects(wrap(&context));

  OpBuilder builder(&context);

  SmallVector<Value> operands;
  for (const auto& [dim_slice, type] : llvm::zip(operand_dims, operand_types)) {
    auto tensor_type = tensat::CostModel::newTensorType(builder, dim_slice, type);
    operands.push_back(OperationTimer::getDummyOp(builder, tensor_type)->getResult(0));
  }

  std::vector<std::vector<int64_t>> other_vecs;
  for (const auto& vec : other_vector_args)
    other_vecs.push_back(rust_slice_to_cpp_vector(vec));

  Operation* mlirOp = nullptr;
  std::vector<int64_t> permutation_vec, shape_vec, lhs_batch_dim, rhs_batch_dim, lhs_contract_dim, rhs_contract_dim, precision_config, shape;
  std::vector<int64_t> start_indices, limit_indices, strides;
  std::vector<Attribute> precisionVec;
  

  switch (op) {
    case tensat::Ops::AddOp:
      mlirOp = builder.create<stablehlo::AddOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::MulOp:
      mlirOp = builder.create<stablehlo::MulOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::DivOp:
      mlirOp = builder.create<stablehlo::DivOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::SubtractOp:
      mlirOp = builder.create<stablehlo::SubtractOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::NegOp:
      mlirOp = builder.create<stablehlo::NegOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::TanhOp:
      mlirOp = builder.create<stablehlo::TanhOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::ExpOp:
      mlirOp = builder.create<stablehlo::ExpOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::MinOp:
      mlirOp = builder.create<stablehlo::MinOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::MaxOp:
      mlirOp = builder.create<stablehlo::MaxOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::TransposeOp:
      mlirOp = builder.create<stablehlo::TransposeOp>(builder.getUnknownLoc(), operands[0], other_vecs[0]);
      break;
    case tensat::Ops::ReshapeOp:
      mlirOp = builder.create<stablehlo::ReshapeOp>(builder.getUnknownLoc(), deriveOutputType(operands[0], other_vecs[0]), operands[0]);
      break;
    case tensat::Ops::DotGeneralOp:
      lhs_batch_dim = other_vecs[0];
      rhs_batch_dim = other_vecs[1];
      lhs_contract_dim = other_vecs[2];
      rhs_contract_dim = other_vecs[3];
      precision_config = other_vecs[4];
      shape = other_vecs[5];

      for (auto& precision : precision_config) {
        switch (precision) {
          case 0:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(&context, stablehlo::Precision::DEFAULT)); break;
          case 1:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(&context, stablehlo::Precision::HIGH)); break;
          case 2:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(&context, stablehlo::Precision::HIGHEST)); break;
        }
      }

      mlirOp = builder.create<stablehlo::DotGeneralOp>(
        builder.getUnknownLoc(),
        deriveOutputType(operands[1], shape),
        operands[0],
        operands[1],
        stablehlo::DotDimensionNumbersAttr::get(&context, lhs_batch_dim, rhs_batch_dim, lhs_contract_dim, rhs_contract_dim),
        mlir::ArrayAttr::get(&context, llvm::ArrayRef(precisionVec))
      );
      break;
    case tensat::Ops::SliceOp:
      start_indices = other_vecs[0];
      limit_indices = other_vecs[1];
      strides = other_vecs[2];
      mlirOp = builder.create<stablehlo::SliceOp>(
        builder.getUnknownLoc(),
	operands[0], 
	start_indices,
	limit_indices,
	strides
      );
      break;
    default:
      std::cout << "EGRAPH INVALID, UNSUPPORTED OP SHAPE REQUESTED" << "\n";
      assert(false);
      return {};
  }

  rust::Vec<tensat::Shape> shapes;
  
  for (auto res : mlirOp->getResults()) {
    auto output_tensor = res.getType().cast<TensorType>();
    auto shape_array = castArrayRefToInt32(output_tensor.getShape());
    rust::Vec<int> rusty_shape;

    for (const auto& dim : shape_array)
      rusty_shape.push_back(dim);

    shapes.push_back({rusty_shape});
  }

  mlirOp->erase();
  return shapes;
}

std::unique_ptr<tensat::ShapeInference> tensat::newShapeInference() {
  return std::make_unique<tensat::ShapeInference>();
}

namespace {
  class EqualitySaturationPass
    : public PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {
    public:
      StringRef getArgument() const override { return "equality-saturation-pass"; }
      StringRef getDescription() const override {
      return "Optimizes HLO graph using a Rust-based optimizer";
    }


    rust::Slice<const int32_t> castArrayRefToInt32Slice(llvm::ArrayRef<int64_t> input) {
      auto input_vec = castArrayRefToInt32(input);
      auto input_slice = rust::Slice<const int32_t>{
        input_vec.data(), static_cast<size_t>(input_vec.size())};
      return input_slice;
    }

    int getValueIndex(Operation* definingOp, Value &value) {
      auto results = definingOp->getResults();
      for (int i = 0; i < results.size(); i++) {
        if (results[i] == value) return i;
      }
      return -1;
    }

    tensat::TensorInfo* handleOperand(
        Value &operand,
        std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
        std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
        std::vector<Operation*> *blackboxIDToTensorInfo,
  OpBuilder &builder,
        Box<tensat::CppGraphConverter> &graph) {
      if (auto defOp = operand.getDefiningOp()) {
        // Use existing TensorInfo if already processed
        int index = getValueIndex(defOp, operand);
        assert(index >= 0);
        auto convertedOperand = dfs(defOp, opToTensorInfo, blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph);
        if (index == 0) {
          return convertedOperand;
        } else {
          auto indexOperand = graph->new_index(index, *convertedOperand).into_raw();
          opToTensorInfo->insert({defOp, indexOperand});
          return indexOperand;
        }
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
          assert(false);
        }
      }
      std::cout
        << "EqualitySaturationPass: encountered operand that is neither the result of an Op nor a BlockArgument."
        << "\n";
      assert(false);
    }

// // Handle integral types, simply return the operand as-is
// template<typename T>
// typename std::enable_if<std::is_integral<T>::value, T>::type
// handleOperand(T operand, std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
//               std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
//               std::vector<Operation*> *blackboxIDToTensorInfo,
//               OpBuilder &builder, 
//               Box<tensat::CppGraphConverter> &graph) {
//     return operand;
// }

// Handle rust::Slice<const int> directly, assuming we can pass it through without modification
std::unique_ptr<rust::Slice<int>> handleOperand(
    std::unique_ptr<rust::Slice<int>> operand,
    std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
    std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
    std::vector<Operation*> *blackboxIDToTensorInfo,
    OpBuilder &builder, 
    Box<tensat::CppGraphConverter> &graph) {
    return operand;
}

    template <typename CreateOpFunc, typename... Args>
    tensat::TensorInfo* handleOperation(
        Operation* op,
        CreateOpFunc createOpFunc,
        std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
        std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
        std::vector<Operation*> *blackboxIDToTensorInfo,
        OpBuilder &builder,
        Box<tensat::CppGraphConverter> &graph,
        Args&&... args) {
      auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
      auto handleArgs = [&](auto&&... operands) {
        return std::make_tuple(handleOperand(operands, opToTensorInfo, blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph)...);
      };

      auto operandInfos = std::apply(handleArgs, args_tuple);

      // Use std::apply to unpack operandInfos into the function call
      return std::apply([&](auto&&... unpacked) {
        return std::invoke(createOpFunc, *graph, *unpacked...).into_raw();
      }, operandInfos);
    }

    tensat::TensorInfo *dfs(Operation* op,
      std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
      std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
      std::vector<Operation*> *blackboxIDToTensorInfo,
      OpBuilder &builder,
      Box<tensat::CppGraphConverter> &graph) {
      // std::cout << "DFS AT " << op->getName().getStringRef().str() << "\n";

      if (opToTensorInfo->find(op) != opToTensorInfo->end()) {
        return opToTensorInfo->at(op);
      }
      tensat::TensorInfo *tensorInfo = nullptr;
      auto handleOperandPartial = [&](auto operand) {
        return handleOperand(operand, opToTensorInfo, blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph); 
      };
      // auto handleOperationPartial = [&](auto&& createOpFunc, auto&&... operands) {
      //   return handleOperation(op, createOpFunc, opToTensorInfo, blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph, std::forward<decltype(operands)>(operands)...);
      // }; 

      if (isa<stablehlo::ConstantOp>(op)) {
        auto constant = cast<stablehlo::ConstantOp>(op);
        auto output_tensor = constant->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };

        tensorInfo = graph->new_constant_op(shape).into_raw();
      } else if (isa<stablehlo::MulOp>(op)) {
        auto mul = cast<stablehlo::MulOp>(op);
        auto output_tensor = mul->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_mul_op(*handleOperandPartial(mul.getLhs()), *handleOperandPartial(mul.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::SubtractOp>(op)) {
        auto subtract = cast<stablehlo::SubtractOp>(op);
        auto output_tensor = subtract->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_subtract_op(*handleOperandPartial(subtract.getLhs()), *handleOperandPartial(subtract.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::DivOp>(op)) {
        auto div = cast<stablehlo::DivOp>(op);
        auto output_tensor = div->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_div_op(*handleOperandPartial(div.getLhs()), *handleOperandPartial(div.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::AddOp>(op)) {
        auto add = cast<stablehlo::AddOp>(op);
        auto output_tensor = add->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_add_op(*handleOperandPartial(add.getLhs()), *handleOperandPartial(add.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::MinOp>(op)) {
        auto min = cast<stablehlo::MinOp>(op);
        auto output_tensor = min->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_min_op(*handleOperandPartial(min.getLhs()), *handleOperandPartial(min.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::MaxOp>(op)) {
        auto max = cast<stablehlo::MaxOp>(op);
        auto output_tensor = max->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_max_op(*handleOperandPartial(max.getLhs()), *handleOperandPartial(max.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::TanhOp>(op)) {
        auto tanh = cast<stablehlo::TanhOp>(op);
        auto output_tensor = tanh->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_tanh_op(*handleOperandPartial(tanh.getOperand()), shape).into_raw();
      } else if (isa<stablehlo::NegOp>(op)) {
        auto neg = cast<stablehlo::NegOp>(op);
        auto output_tensor = neg->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_neg_op(*handleOperandPartial(neg.getOperand()), shape).into_raw();
      } else if (isa<stablehlo::ExpOp>(op)) {
        auto exp = cast<stablehlo::ExpOp>(op);
        auto output_tensor = exp->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_exp_op(*handleOperandPartial(exp.getOperand()), shape).into_raw();
      } else if (isa<stablehlo::TransposeOp>(op)) {
        auto transpose = cast<stablehlo::TransposeOp>(op);
        std::vector<int32_t> permutation = castArrayRefToInt32(transpose.getPermutation());
        auto permutation_slice = rust::Slice<const int> {
          permutation.data(), static_cast<size_t>(permutation.size())};
        auto output_shape = castArrayRefToInt32(transpose->getResult(0).getType().cast<TensorType>().getShape());
	auto output_shape_slice = rust::Slice<const int> {
		output_shape.data(), output_shape.size() };
        tensorInfo = graph->new_transpose_op(
          *handleOperandPartial(transpose.getOperand()),
          permutation_slice,
          output_shape_slice
        ).into_raw();
      } else if (isa<stablehlo::ReshapeOp>(op)) {
        auto reshape = cast<stablehlo::ReshapeOp>(op);
        if (auto output_tensor = reshape.getResult().getType().cast<TensorType>()) {
          auto shape = castArrayRefToInt32(output_tensor.getShape());
          auto output_shape_slice = rust::Slice<const int> {
            shape.data(), shape.size()};
          tensorInfo = graph->new_reshape_op(
            *handleOperandPartial(reshape.getOperand()),
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
          auto output_shape_slice = rust::Slice<const int>{
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
        auto lhs_batch_dim = castArrayRefToInt32(dot_dim_attrs.getLhsBatchingDimensions());
        auto rhs_batch_dim = castArrayRefToInt32(dot_dim_attrs.getRhsBatchingDimensions());
        auto lhs_contract_dim = castArrayRefToInt32(dot_dim_attrs.getLhsContractingDimensions());
        auto rhs_contract_dim = castArrayRefToInt32(dot_dim_attrs.getRhsContractingDimensions());
       
        mlir::ArrayAttr precision = dot_general.getPrecisionConfig().value_or(mlir::ArrayAttr());
        std::vector<int> precision_configs;
        for (int i = 0; i < precision.size(); i++) {
          auto precisionAttr = precision[i].dyn_cast<mlir::stablehlo::PrecisionAttr>();
          if (!precisionAttr) continue;  // Skip if it's not a PrecisionAttr, although such attributes should not exist here
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
          precision_configs.data(), precision_configs.size()};

        if (auto output_tensor = dot_general.getResult().getType().cast<TensorType>()) {
          auto shape = castArrayRefToInt32(output_tensor.getShape());
          auto output_shape_slice = rust::Slice<const int> {
            shape.data(), shape.size()};

          tensorInfo = graph->new_dot_general_op(
            *handleOperandPartial(dot_general.getLhs()),
            *handleOperandPartial(dot_general.getRhs()),
            {lhs_batch_dim.data(), lhs_batch_dim.size()},
            {rhs_batch_dim.data(), rhs_batch_dim.size()},
            {lhs_contract_dim.data(), lhs_contract_dim.size()},
            {rhs_contract_dim.data(), rhs_contract_dim.size()},
            precision_config_slice,
            output_shape_slice
          ).into_raw();
        } else {
          std::cout << "EqualitySaturationPass: result of stablehlo::DotGeneralOp has non-tensor type" << std::endl;
        }
      } else if (isa<stablehlo::ConcatenateOp>(op)) {
        auto concat = cast<stablehlo::ConcatenateOp>(op);
        auto output_tensor = concat->getResult(0).getType().cast<TensorType>();
        auto output_shape_array = castArrayRefToInt32(output_tensor.getShape());
        std::vector<tensat::TensorInfo*> inputs;
        for (auto input : concat.getInputs()) {
          inputs.push_back(handleOperandPartial(input));
        }
        int32_t dimension = concat.getDimension();
        tensorInfo = graph->new_concatenate_op(
          { inputs.data(), inputs.size() },
          dimension,
          { output_shape_array.data(), output_shape_array.size() }
        ).into_raw();
      } else if (isa<stablehlo::SliceOp>(op)) {
        auto slice = cast<stablehlo::SliceOp>(op);
        auto output_tensor = slice->getResult(0).getType().cast<TensorType>();
        auto output_shape_array = castArrayRefToInt32(output_tensor.getShape());
        std::vector<tensat::TensorInfo*> inputs;
        auto operand = handleOperandPartial(slice.getOperand());
        auto start_indices = castArrayRefToInt32(slice.getStartIndices());
        auto limit_indices = castArrayRefToInt32(slice.getLimitIndices());
        auto strides = castArrayRefToInt32(slice.getStrides());
        tensorInfo = graph->new_slice_op(
          *operand,
          { start_indices.data(), start_indices.size() },
          { limit_indices.data(), limit_indices.size() },
          { strides.data(), strides.size() },
          { output_shape_array.data(), output_shape_array.size() }
        ).into_raw();
      } else {
        int numOperands = op->getNumOperands();
        std::vector<rust::Slice<const int>> shapes_vec;
        for (auto result : op->getResults()) {
          auto output_tensor = result.getType().cast<TensorType>();
          auto shape_array = castArrayRefToInt32(output_tensor.getShape());
          shapes_vec.push_back({shape_array.data(), shape_array.size()});
        }
        rust::Slice<const rust::Slice<const int>> shapes = {shapes_vec.data(), shapes_vec.size()};
        auto output_tensor = op->getResult(0).getType().cast<TensorType>();
        std::vector<tensat::TensorInfo*> processedOperands;
        auto copy = OperationTimer::cloneOpInContext(builder, op);
        // auto copy = op->clone();
        blackboxIDToTensorInfo->push_back(copy);
        int blackboxOpID = blackboxIDToTensorInfo->size()-1;
        for (size_t i = 0; i < numOperands; i++) {
          auto operand = handleOperandPartial(op->getOperand(i));
          processedOperands.push_back(operand);
        }
        auto operandPtrsSlice = rust::Slice<tensat::TensorInfo* const>{processedOperands.data(), static_cast<size_t>(processedOperands.size())};
        tensorInfo = graph->new_blackbox_op(
          operandPtrsSlice,
          blackboxOpID,
          shapes
        ).into_raw();
      }
      if (tensorInfo != nullptr) {
        opToTensorInfo->insert({op, tensorInfo});
        return tensorInfo;
      }
      return tensorInfo;
    }

    Box<tensat::CppGraphConverter> createEgraph(
        std::vector<Operation*> *blackboxIDToTensorInfo,
        OpBuilder &builder,
        ModuleOp module) {

      auto graph = tensat::new_converter();
      // members of the class
      std::unordered_map<Operation*, tensat::TensorInfo*> opToTensorInfo;
      std::unordered_map<int, tensat::TensorInfo*> blockArgToTensorInfo;

      module.walk([&](func::ReturnOp op) {
        // Call dfs() on the things that are returned.
        for (auto value : op.getOperands()) {
          dfs(value.getDefiningOp(), &opToTensorInfo, &blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph);
        }
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

    /**
     * Parse the Vec nodes with Nums (e.g Vec(Num(128), Num(128))) emitted by tensat node construction.
     */
    std::vector<int64_t> parseNumVec(rust::vec<tensat::Node> &nodes, tensat::Node &seq) {
      std::vector<int64_t> result;
      
      for (auto i : seq.operands) {
        result.push_back(nodes[i].operands[0]);
      }

      return result;
    }

    /**
     * Parse the Vec nodes with arbitrary operations (e.g Vec(Input(...), AddOp(...))) emitted by tensat node construction.
     */
    std::vector<Value> parseOpVec(std::vector<Value> &opVals, tensat::Node &seq) {
      std::vector<Value> result;

      for (auto i : seq.operands) {
        result.push_back(opVals[i]);
      }
      
      return result;
   }


    void reconstructStablehlo(ModuleOp *root, std::vector<Operation*> *blackboxIDToTensorInfo, rust::vec<tensat::Node> &nodes, OpBuilder &builder) {
      auto context = root->getContext();
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
        if (node.name == "Var" || node.name == "Num") {
          /* do nothing */
        } else if (node.name == "Input") {
          int blockArgNumber = nodes[node.operands[1]].operands[0];
          opVals.push_back(block.getArgument(blockArgNumber));
          continue;
        } else if (node.name == "Index") {
          int index = nodes[node.operands[1]].operands[0];
          opVals.push_back(opVals[index].getDefiningOp()->getResult(index));
          continue;
        } else if (node.name == "NegOp") {
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
        } else if (node.name == "TransposeOp") {
          auto input = opVals[node.operands[0]];
          // TODO: Can this be done cleaner, getting the Value earlier from Var instead of accessing now?
          //  Can't be certain until we've implemented many ops using Var.
          // TODO: Untested
          auto permutation = parseNumVec(nodes, nodes[node.operands[1]]);
          newOp = builder.create<stablehlo::TransposeOp>(location, input, permutation);
        } else if (node.name == "ReshapeOp") {
          // TODO: Untested
          auto input = opVals[node.operands[0]];
          auto shape = parseNumVec(nodes, nodes[node.operands[1]]);
          auto newType = deriveOutputType(input, shape);
          newOp = builder.create<stablehlo::ReshapeOp>(location, newType, input);
        } else if (node.name == "DotGeneralOp") {
          // TODO: Untested
          auto lhs = opVals[node.operands[0]];
          auto rhs = opVals[node.operands[1]];

          auto lhsBatchDim = parseNumVec(nodes, nodes[node.operands[2]]);
          auto rhsBatchDim = parseNumVec(nodes, nodes[node.operands[3]]);
          auto lhsContractDim = parseNumVec(nodes, nodes[node.operands[4]]);
          auto rhsContractDim = parseNumVec(nodes, nodes[node.operands[5]]);
          auto precisionConfig = parseNumVec(nodes, nodes[node.operands[6]]);
          auto shape = parseNumVec(nodes, nodes[node.operands[7]]);

          auto dotDimensionNumbersAttr = stablehlo::DotDimensionNumbersAttr::get(context, lhsBatchDim, rhsBatchDim, lhsContractDim, rhsContractDim);
          
          std::vector<Attribute> precisionVec;

          for (auto& precision : precisionConfig) {
            switch (precision) {
              case 0:
                precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::DEFAULT)); break;
              case 1:
                precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::HIGH)); break;
              case 2:
                precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::HIGHEST)); break;
            }
          }

          // TODO: Is lhs correct here?
          auto newType = deriveOutputType(lhs, shape);
          newOp = builder.create<stablehlo::DotGeneralOp>(location, newType, lhs, rhs, dotDimensionNumbersAttr, mlir::ArrayAttr::get(context, llvm::ArrayRef(precisionVec)));
        } else if (node.name == "ConcatenateOp") {
          auto inputs = parseOpVec(opVals, nodes[node.operands[0]]);
          auto dimension = nodes[node.operands[1]].operands[0];
          newOp = builder.create<stablehlo::ConcatenateOp>(location, inputs, dimension);
        } else if (node.name == "SliceOp") {
          auto operand = opVals[node.operands[0]];
          auto startIndices = parseNumVec(nodes, nodes[node.operands[1]]);
          auto limitIndices = parseNumVec(nodes, nodes[node.operands[2]]);
          auto strides = parseNumVec(nodes, nodes[node.operands[3]]);
          newOp = builder.create<stablehlo::SliceOp>(location, operand, startIndices, limitIndices, strides);
        } else if (node.name == "blackbox") {
          size_t numOperands = node.operands.size() - 1;
          auto blackboxID = nodes[node.operands[numOperands]].operands[0];
          Operation* newOp = blackboxIDToTensorInfo->at(blackboxID);
    
          // Really subtle error arose here from not handling Num properly.
          // We might want to have a Num hashmap 
          for (size_t i = 0; i < numOperands; ++i) {
            auto operandIndex = node.operands[i];
            auto operand = opVals[operandIndex];
            newOp->setOperand(i, operand);
          }
    
          // Do we need to account for insertion points at all?
          builder.insert(newOp);
    
          // TODO: why does everything break when we comment these four lines below?
          block.push_back(newOp);
          opVals.push_back(newOp->getResult(0));
          std::cout << "pushed to block" << "\n";
          continue;
        } else {
          // TODO: implement other operations
          std::cout << "UNIMPLEMENTED " << node.name << "\n";
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

      auto returnOp = builder.create<func::ReturnOp>(builder.getUnknownLoc(),
      block.back().getResults()); block.push_back(returnOp);
    }

    void runOnOperation() override {
      ModuleOp module = getOperation();
      std::cout << "ORIGINAL MODULE" << "\n";
      module.dump();
      std::vector<Operation*> blackboxIDToTensorInfo;
      auto context = module->getContext();
      OpBuilder builder(context);
      auto graph = createEgraph(&blackboxIDToTensorInfo, builder, module);
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
      reconstructStablehlo(&module, &blackboxIDToTensorInfo, optimized, builder);
      std::cout << "SUPEROPTIMIZED MODULE" << "\n";
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
