/*
* SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
class AllocOpPattern : public mlir::OpRewritePattern<memref::AllocOp>
{
public:
  AllocOpPattern(mlir::MLIRContext* context, size_t& globalsCounter)
      : mlir::OpRewritePattern<memref::AllocOp>(context),
        globalsCounter(&globalsCounter)
  {
  }

  mlir::LogicalResult matchAndRewrite(
      memref::AllocOp op,
      mlir::PatternRewriter& rewriter) const override
  {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto globalOp = rewriter.create<memref::GlobalOp>(
        op->getLoc(), "alloc" + std::to_string((*globalsCounter)++),
        rewriter.getStringAttr("private"),
        op.getType(),
        ElementsAttr(), false, nullptr);

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(
        op, op.getType(), globalOp.getSymName());

    return mlir::success();
  }

private:
  size_t* globalsCounter;
};

class DeallocOpPattern : public mlir::OpRewritePattern<memref::DeallocOp>
{
public:
  using mlir::OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      memref::DeallocOp op,
      mlir::PatternRewriter& rewriter) const override
  {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};
}

namespace {
class ConvertAllocsToGlobalsPass : public PassWrapper<ConvertAllocsToGlobalsPass,
                                       OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAllocsToGlobalsPass)

  ConvertAllocsToGlobalsPass()
  {
  }

  StringRef getArgument() const override { return "convert-allocs-to-globals"; }

  StringRef getDescription() const override {
    return "Convert the allocs to global declarations.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp);
    size_t globalsCounter = 0;

    mlir::ConversionTarget target(getContext());

    target.addDynamicallyLegalOp<memref::AllocOp>([](memref::AllocOp op) {
      return !op.getType().hasStaticShape();
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([](memref::DeallocOp op) {
      return !op.getMemref().getType().hasStaticShape();
    });

    target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
      return true;
    });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<AllocOpPattern>(&getContext(), globalsCounter);
    patterns.insert<DeallocOpPattern>(&getContext());

    if (mlir::failed(applyPartialConversion(
            moduleOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}

namespace onnx_mlir {
std::unique_ptr<Pass> createConvertAllocsToGlobalsPass() {
  return std::make_unique<ConvertAllocsToGlobalsPass>();
}
}
