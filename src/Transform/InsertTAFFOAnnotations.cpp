/*
* SPDX-License-Identifier: Apache-2.0
*/

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
  class InsertTAFFOAnnotationsPass : public PassWrapper<InsertTAFFOAnnotationsPass,
                                      OperationPass<ModuleOp>> {
  private:
    double lowerBound;
    double upperBound;

  public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertTAFFOAnnotationsPass)

   InsertTAFFOAnnotationsPass(double lowerBound, double upperBound)
      : lowerBound(lowerBound), upperBound(upperBound)
   {
   }

   StringRef getArgument() const override { return "insert-taffo-annotations"; }

   StringRef getDescription() const override {
     return "Insert the annotations for TAFFO.";
   }

   void runOnOperation() override {
     auto moduleOp = getOperation();
     OpBuilder builder(moduleOp);
     size_t annotationsCounter = 0;

     if (failed(annotateMainGraphFunction(
             builder, moduleOp, annotationsCounter))) {
       return signalPassFailure();
     }

     /*
     if (failed(annotateMainGraphInterfaceFunction(
             builder, moduleOp, annotationsCounter))) {
       return signalPassFailure();
     }
      */
   }

   LogicalResult annotateMainGraphFunction(
       OpBuilder& builder,
       ModuleOp moduleOp,
       size_t& annotationsCounter);

   LogicalResult annotateMainGraphFunctionArgs(
       OpBuilder& builder,
       LLVM::LLVMFuncOp funcOp,
       size_t& annotationsCounter);

   LogicalResult annotateMainGraphFunctionMallocs(
       OpBuilder& builder,
       LLVM::LLVMFuncOp funcOp,
       size_t& annotationsCounter);

   /*
   LogicalResult annotateMainGraphInterfaceFunction(
       OpBuilder& builder,
       ModuleOp moduleOp,
       size_t& annotationsCounter);

   LLVM::LLVMFuncOp createMediatorFunction(
       OpBuilder& builder,
       ModuleOp moduleOp,
       Location loc,
       size_t& mediatorsCounter,
       ArrayRef<Type> inputTypes,
       Type resultType,
       size_t& annotationsCounter);
       */

   Type getVoidPtrType();

   Value getOrCreateGlobalString(
       OpBuilder& builder,
       Location loc,
       ModuleOp moduleOp,
       StringRef name,
       StringRef value);

   /*
   LLVM::LLVMFuncOp getOrDeclareVarAnnotationFunction(
       OpBuilder& builder, ModuleOp moduleOp);

   LogicalResult createVarAnnotation(
       OpBuilder& builder, ModuleOp moduleOp, Location loc,
       Value ptr, Value annotation, int lineNumber);
   */
  };
} // namespace

LogicalResult InsertTAFFOAnnotationsPass::annotateMainGraphFunction(
    OpBuilder& builder, ModuleOp moduleOp,
    size_t& annotationsCounter)
{
   auto funcOp =
       moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");

   if (!funcOp) {
     return failure();
   }

   if (failed(annotateMainGraphFunctionArgs(
           builder, funcOp, annotationsCounter))) {
     return failure();
   }

   if (failed(annotateMainGraphFunctionMallocs(
           builder, funcOp, annotationsCounter))) {
     return failure();
   }

   return success();
}

LogicalResult InsertTAFFOAnnotationsPass::annotateMainGraphFunctionArgs(
    OpBuilder& builder,
    LLVM::LLVMFuncOp funcOp,
    size_t& annotationsCounter)
{
   OpBuilder::InsertionGuard guard(builder);
   builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());

   Location loc = funcOp.getLoc();
   auto moduleOp = funcOp->getParentOfType<ModuleOp>();

   auto annotatePtrFn = [&](Value ptr, std::string targetStr) -> LogicalResult {
     if (ptr.getType() != getVoidPtrType()) {
       ptr = builder.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr);
     }

     Value annotation = getOrCreateGlobalString(
         builder, ptr.getLoc(), moduleOp,
         "annotation_" + std::to_string(annotationsCounter++),
         targetStr + " scalar(range(" + std::to_string(lowerBound) + ", " +
             std::to_string(upperBound) + ") final disabled)");

     Value lineNumber = builder.create<LLVM::ConstantOp>(
         loc, builder.getI32IntegerAttr(0));

     Value nullptrValue = builder.create<LLVM::NullOp>(
         loc, getVoidPtrType());

     Value fileNameValue = getOrCreateGlobalString(
         builder, moduleOp.getLoc(), moduleOp, "fileName", "");

     builder.create<LLVM::VarAnnotation>(
         loc, ptr, annotation, fileNameValue, lineNumber, nullptrValue);

     return success();
   };

   if (failed(annotatePtrFn(funcOp.getArgument(0), "target('onnx')"))) {
     return failure();
   }

   if (failed(annotatePtrFn(funcOp.getArgument(1), "target('onnx')"))) {
     return failure();
   }

   return success();
}

static void collectMallocs(
    LLVM::LLVMFuncOp funcOp,
    SmallVectorImpl<LLVM::CallOp>& mallocCalls)
{
   funcOp.walk([&](LLVM::CallOp callOp) {
     if (auto callee = callOp.getCallee(); callee && *callee == "malloc") {
       mallocCalls.push_back(callOp);
     }
   });
}

static void collectAllocaOps(
    LLVM::LLVMFuncOp funcOp,
    SmallVectorImpl<LLVM::AllocaOp>& allocaOps)
{
   funcOp.walk([&](LLVM::AllocaOp allocaOp) {
     if (allocaOp.getType().getElementType().isIntOrFloat()) {
       allocaOps.push_back(allocaOp);
     }
   });
}

LogicalResult InsertTAFFOAnnotationsPass::annotateMainGraphFunctionMallocs(
    OpBuilder& builder,
    LLVM::LLVMFuncOp funcOp,
    size_t& annotationsCounter)
{
   OpBuilder::InsertionGuard guard(builder);
   auto moduleOp = funcOp->getParentOfType<ModuleOp>();

   SmallVector<LLVM::CallOp> mallocCalls;
   collectMallocs(funcOp, mallocCalls);

   for (LLVM::CallOp callOp : mallocCalls) {
     builder.setInsertionPointAfter(callOp);

     Value ptr = callOp.getResult();
     Location loc = ptr.getLoc();

     Value annotation = getOrCreateGlobalString(
         builder, ptr.getLoc(), moduleOp,
         "annotation_" + std::to_string(annotationsCounter++),
         "scalar(range(" + std::to_string(lowerBound) + ", " +
             std::to_string(upperBound) + ") final disabled)");

     Value lineNumber = builder.create<LLVM::ConstantOp>(
         loc, builder.getI32IntegerAttr(0));

     Value nullptrValue = builder.create<LLVM::NullOp>(
         loc, getVoidPtrType());

     Value fileNameValue = getOrCreateGlobalString(
         builder, loc, moduleOp, "fileName", "");

     builder.create<LLVM::VarAnnotation>(
         loc, ptr, annotation, fileNameValue, lineNumber, nullptrValue);
   }

   SmallVector<LLVM::AllocaOp> allocaOps;
   collectAllocaOps(funcOp, allocaOps);

   for (LLVM::AllocaOp allocaOp : allocaOps) {
     builder.setInsertionPointAfter(allocaOp);

     Value ptr = allocaOp.getResult();
     Location loc = ptr.getLoc();

     if (ptr.getType() != getVoidPtrType()) {
       ptr = builder.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr);
     }

     Value annotation = getOrCreateGlobalString(
         builder, ptr.getLoc(), moduleOp,
         "annotation_" + std::to_string(annotationsCounter++),
         "scalar(range(" + std::to_string(lowerBound) + ", " +
             std::to_string(upperBound) + ") final disabled)");

     Value lineNumber = builder.create<LLVM::ConstantOp>(
         loc, builder.getI32IntegerAttr(0));

     Value nullptrValue = builder.create<LLVM::NullOp>(
         loc, getVoidPtrType());

     Value fileNameValue = getOrCreateGlobalString(
         builder, loc, moduleOp, "fileName", "");

     builder.create<LLVM::VarAnnotation>(
         loc, ptr, annotation, fileNameValue, lineNumber, nullptrValue);
   }

   return success();
}

/*
static void getMainGraphCallOp(
    LLVM::LLVMFuncOp funcOp, llvm::SmallVectorImpl<LLVM::CallOp>& callOps)
{
   for (LLVM::CallOp callOp : funcOp.getOps<LLVM::CallOp>()) {
     if (auto callee = callOp.getCallee();
         callee && callee == "main_graph") {
       callOps.push_back(callOp);
     }
   }
}

LogicalResult
InsertTAFFOAnnotationsPass::annotateMainGraphInterfaceFunction(
    OpBuilder& builder, ModuleOp moduleOp,
    size_t& annotationsCounter)
{
   auto funcOp =
       moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("_mlir_ciface_main_graph");

   if (!funcOp) {
     return failure();
   }

   OpBuilder::InsertionGuard guard(builder);
   builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());

   llvm::SmallVector<LLVM::CallOp> callOps;
   getMainGraphCallOp(funcOp, callOps);

   size_t mediatorsCounter = 0;

   for (LLVM::CallOp callOp : callOps) {
     builder.setInsertionPoint(callOp);

     SmallVector<Type> inputTypes;

     for (Value callArg : callOp.getOperands()) {
       inputTypes.push_back(callArg.getType());
     }

     LLVM::LLVMFuncOp mediatorFunction = createMediatorFunction(
         builder, moduleOp, callOp.getLoc(), mediatorsCounter,
         inputTypes, callOp.getResult().getType(), annotationsCounter);

     auto newCallOp = builder.create<LLVM::CallOp>(
         callOp.getLoc(), mediatorFunction, callOp.getOperands());

     callOp.getResult().replaceAllUsesWith(newCallOp.getResult());
     callOp.erase();
   }

   return success();
}
*/

/*
LLVM::LLVMFuncOp InsertTAFFOAnnotationsPass::createMediatorFunction(
    OpBuilder& builder, ModuleOp moduleOp, Location loc,
    size_t& mediatorsCounter,
    ArrayRef<Type> inputTypes, Type resultType,
    size_t& annotationsCounter)
{
   OpBuilder::InsertionGuard guard(builder);
   builder.setInsertionPointToStart(moduleOp.getBody());

   auto funcOp = builder.create<LLVM::LLVMFuncOp>(
       loc, "mediator_" + std::to_string(mediatorsCounter++),
       LLVM::LLVMFunctionType::get(resultType, inputTypes));

   Block* entryBlock = funcOp.addEntryBlock();
   builder.setInsertionPointToStart(entryBlock);

   Value ptr = funcOp.getArgument(0);

   if (ptr.getType() != getVoidPtrType()) {
     ptr = builder.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr);
   }

   Value annotation = getOrCreateGlobalString(
       builder, ptr.getLoc(), moduleOp,
       "annotation_" + std::to_string(annotationsCounter++),
       "scalar(range(-3000, 3000))");

   Value lineNumber = builder.create<LLVM::ConstantOp>(
       loc, builder.getI32IntegerAttr(0));

   Value nullptrValue = builder.create<LLVM::NullOp>(
       loc, getVoidPtrType());

   //if (failed(createVarAnnotation(
   //        builder, moduleOp, ptr.getLoc(), ptr, annotation, 0))) {
   //  return failure();
   //}

   Value fileNameValue = getOrCreateGlobalString(
       builder, moduleOp.getLoc(), moduleOp,
       "fileName", "");

   builder.create<LLVM::VarAnnotation>(
       loc, ptr, annotation, fileNameValue, lineNumber, nullptrValue);

   auto callOp = builder.create<LLVM::CallOp>(
       loc, resultType, "main_graph", funcOp.getArguments());

   builder.create<LLVM::ReturnOp>(loc, callOp.getResults());
   return funcOp;
}
 */

Type InsertTAFFOAnnotationsPass::getVoidPtrType()
{
   Type i8Type = IntegerType::get(&getContext(), 8);
   return LLVM::LLVMPointerType::get(i8Type);
}

Value InsertTAFFOAnnotationsPass::getOrCreateGlobalString(
    OpBuilder& builder,
    Location loc,
    ModuleOp moduleOp,
    StringRef name,
    StringRef value)
{
   LLVM::GlobalOp global;

   if (auto existingString = moduleOp.lookupSymbol<LLVM::GlobalOp>(name)) {
     global = existingString;
   } else {
     // Create the global at the entry of the module.
     OpBuilder::InsertionGuard insertGuard(builder);
     builder.setInsertionPointToStart(moduleOp.getBody());

     auto type = LLVM::LLVMArrayType::get(
         IntegerType::get(builder.getContext(), 8), value.size() + 1);

     global = builder.create<LLVM::GlobalOp>(loc, type, true,
         LLVM::Linkage::Internal, name,
         builder.getStringAttr(
             llvm::StringRef(value.data(), value.size() + 1)));
   }

   // Get the pointer to the first character of the global string.
   Value globalPtr =
       builder.create<LLVM::AddressOfOp>(loc, global);

   Value cst0 = builder.create<LLVM::ConstantOp>(loc,
       IntegerType::get(builder.getContext(), 64),
       builder.getIntegerAttr(builder.getIndexType(), 0));

   return builder.create<LLVM::GEPOp>(
       loc, getVoidPtrType(), globalPtr, llvm::ArrayRef({cst0, cst0}));
}

/*
LLVM::LLVMFuncOp InsertTAFFOAnnotationsPass::getOrDeclareVarAnnotationFunction(
    OpBuilder& builder, ModuleOp moduleOp)
{
   if (auto funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(
           "llvm.var.annotation")) {
     return funcOp;
   }

   OpBuilder::InsertionGuard guard(builder);
   builder.setInsertionPointToStart(moduleOp.getBody());

   llvm::SmallVector<Type> inputTypes;
   inputTypes.push_back(getVoidPtrType());
   inputTypes.push_back(getVoidPtrType());
   inputTypes.push_back(getVoidPtrType());
   inputTypes.push_back(builder.getI32Type());
   inputTypes.push_back(getVoidPtrType());

   auto functionType = LLVM::LLVMFunctionType::get(
       LLVM::LLVMVoidType::get(&getContext()), inputTypes);

   return builder.create<LLVM::LLVMFuncOp>(
       moduleOp.getLoc(), "llvm.var.annotation", functionType);
}

LogicalResult InsertTAFFOAnnotationsPass::createVarAnnotation(
    OpBuilder& builder, ModuleOp moduleOp, Location loc,
    Value ptr, Value annotation, int lineNumber)
{
   Value fileNameValue = getOrCreateGlobalString(
       builder, moduleOp.getLoc(), moduleOp,
       "fileName", "");

   Value lineNumberValue = builder.create<LLVM::ConstantOp>(
       loc, builder.getI32IntegerAttr(lineNumber));

   Value nullptrValue =
       builder.create<LLVM::NullOp>(loc, getVoidPtrType());

   llvm::SmallVector<Value> args;
   args.push_back(ptr);
   args.push_back(annotation);
   args.push_back(fileNameValue);
   args.push_back(lineNumberValue);
   args.push_back(nullptrValue);

   builder.create<LLVM::CallOp>(
       loc, getOrDeclareVarAnnotationFunction(builder, moduleOp), args);

   return success();
}
*/

namespace onnx_mlir {
    std::unique_ptr<Pass> createInsertTAFFOAnnotationsPass(
        double lowerBound, double upperBound) {
       return std::make_unique<InsertTAFFOAnnotationsPass>(
            lowerBound, upperBound);
    }
}
