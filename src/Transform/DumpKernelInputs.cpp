/*
* SPDX-License-Identifier: Apache-2.0
 */

//===-------- EnableMemoryPool.cpp - Enable Memory Pool for MemRefs -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// For certain cases the number of individual memory allocations required for
// all internal tensors is large and needs to be mitigated. This pass enables a
// managed memory pool for allocating MemRefs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
  class DumpKernelInputsPass : public PassWrapper<DumpKernelInputsPass,
                                         OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpKernelInputsPass)

    StringRef getArgument() const override { return "dump-kernel-inputs"; }

    StringRef getDescription() const override {
      return "Dump the kernel input values.";
    }

    void runOnOperation() override {
      auto moduleOp = getOperation();

      SymbolTableCollection symbolTableCollection;
      SmallVector<func::FuncOp> entryPoints;

      moduleOp.walk([&](ONNXEntryPointOp op) {
        auto funcOp = symbolTableCollection.lookupSymbolIn<func::FuncOp>(
            moduleOp, op.getFunc());

        if (funcOp) {
          entryPoints.push_back(funcOp);
        }
      });

      for (func::FuncOp entryPoint : entryPoints) {
        if (failed(processEntryPoint(entryPoint))) {
          return signalPassFailure();
        }
      }
    }

    LogicalResult processEntryPoint(func::FuncOp entryPoint);

    LogicalResult processOp(OpBuilder& builder, Operation* op);
  };
} // namespace

LogicalResult DumpKernelInputsPass::processEntryPoint(func::FuncOp entryPoint)
{
  auto returnOp = entryPoint.getFunctionBody().back().getTerminator();

  if (!returnOp) {
    return failure();
  }

  mlir::OpBuilder builder(returnOp);
  return processOp(builder, returnOp);
}

LogicalResult DumpKernelInputsPass::processOp(OpBuilder& builder, Operation* op)
{
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  for (Value operand : op->getOperands()) {
    if (operand.getType().isa<TensorType>())
      builder.create<ONNXPrintOp>(op->getLoc(), operand);
  }

  for (Value operand : op->getOperands()) {
    Operation* definingOp = operand.getDefiningOp();

    if (!definingOp)
      continue;

    if (failed(processOp(builder, definingOp)))
      return failure();
  }

  return success();
}

namespace onnx_mlir {
  std::unique_ptr<Pass> createDumpKernelInputsPass() {
    return std::make_unique<DumpKernelInputsPass>();
  }
}
