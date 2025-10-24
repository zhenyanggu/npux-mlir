#include "Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/npux/ir/npuxDialect.h"
#include "src/Dialect/npux/ir/npuxOps.h"

// Debug helpers
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#define DEBUG_TYPE "outline-npux-ops"

// Generated pass declarations.
#define GEN_PASS_CLASSES
#include "src/Dialect/npux/transforms/npuxPasses.h.inc"

using namespace mlir;

namespace npux {
namespace {
struct OutlineNpuxOpsPass : public OutlineNpuxOpsPassBase<OutlineNpuxOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, npux::npuxDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Ensure func dialect is available for creating functions/calls.
    MLIRContext *ctx = &getContext();
    ctx->loadDialect<func::FuncDialect>();

    SymbolTable symbolTable(module.getOperation());
    unsigned counter = 0; // simple unique suffix per module

    // Snapshot original function list to avoid processing newly created
    // outlined functions and prevent infinite growth/hangs.
    SmallVector<func::FuncOp> originalFuncs;
    for (auto f : module.getOps<func::FuncOp>())
      originalFuncs.push_back(f);

    LLVM_DEBUG(llvm::dbgs() << "[OutlineNpuxOps] Start. Original funcs: "
                            << originalFuncs.size() << "\n");
    module->emitRemark() << "OutlineNpuxOps: start (funcs="
                         << (int64_t)originalFuncs.size() << ")";

    // Attribute to mark generated outlined functions (best-effort tagging).
    auto outlinedTag = StringAttr::get(ctx, "npux.outlined.generated");

    int64_t totalOutlined = 0;

    // Walk all functions in the module.
    // 可能存在问题，应该是找到main函数
    for (func::FuncOp funcOp : originalFuncs) {
      if (!funcOp)
        continue;

      SmallVector<Operation *> toProcess;
      // Collect first to avoid iterator invalidation while mutating.
      funcOp.getOperation()->walk([&](Operation *op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "npux")
          toProcess.push_back(op);
      });

      LLVM_DEBUG(llvm::dbgs() << "[OutlineNpuxOps] In func @"
                              << funcOp.getSymName() << ", npux ops found: "
                              << toProcess.size() << "\n");

      for (Operation *op : toProcess) {
        // Initial version: only outline ops without regions.
        if (op->getNumRegions() != 0)
          continue; // or report/emitRemark; kept simple per requirement

        // Build function type from operand types to result types.
        SmallVector<Type> argTypes;
        argTypes.reserve(op->getNumOperands());
        for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i)
          argTypes.push_back(op->getOperand(i).getType());

        SmallVector<Type> retTypes;
        retTypes.reserve(op->getNumResults());
        for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
          retTypes.push_back(op->getResult(i).getType());

        FunctionType fnType = FunctionType::get(ctx, argTypes, retTypes);

        // Create a unique function name: npux_outlined_<opname>_<id>
        std::string baseName = ("npux_outlined_" + op->getName().getStringRef()).str();
        std::string fnName;
        do {
          fnName = baseName + "_" + std::to_string(counter++);
        } while (symbolTable.lookup<func::FuncOp>(fnName));

        // Create the function op and insert into module.
        OpBuilder modBuilder(ctx);
        OpBuilder::InsertionGuard guard(modBuilder);
        modBuilder.setInsertionPointToEnd(module.getBody());
        auto outlined = modBuilder.create<func::FuncOp>(op->getLoc(), fnName, fnType);
        outlined.setPrivate();
        outlined->setAttr(outlinedTag, UnitAttr::get(ctx));

        // Insert into symbol table to finalize symbol state.
        symbolTable.insert(outlined);

        // Build function body: one block with arguments, clone op, return results.
        Region &body = outlined.getBody();
        auto *entry = new Block();
        body.push_back(entry);
        SmallVector<Location> argLocs(argTypes.size(), op->getLoc());
        entry->addArguments(argTypes, argLocs);
        OpBuilder b(entry, entry->begin());
        
        // 插入npux::mvinOp
        // 对每一个输入参数，创建一个mvinOp，将参数从主存移动到NPU内存
        // nvinOp如下
        // def npux_mvinOp : npux_BaseOp<"mvin"> {
//     let summary = "Memory to NPU transfer operation";

//     let description = [{
//         This operation transfers data from main memory to NPU memory.
//     }];

//     let arguments = (ins
//         AnyTensor:$input,
//         AddressSpaceAttr:$memspace  // 这里用上面定义的枚举类型
//     );

//     let results = (outs
//         AnyTensor:$output
//     );
// }
        SmallVector<Value> mvinResults;
        for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i)
        {
          auto arg = entry->getArgument(i);
          auto mvinOp = b.create<npux::mvinOp>(op->getLoc(), arg.getType(), arg, "NPU_SPM");
          mvinResults.push_back(mvinOp.getResult());
        }

        // 后面也要对应的修改，copy的op的输入改为mvin的输出
        // Map original operands to function arguments
        // IRMapping mapper;
        // for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i)
        // {
        //   mapper.map(op->getOperand(i), entry->getArgument(i));
        // }

        // // Clone the op into the function body with remapped operands.
        // Operation *cloned = op->clone(mapper);
        // cloned->setLoc(op->getLoc());
        // b.insert(cloned);
        // Clone the op into the function body with remapped operands.
        IRMapping mapper;
        for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i)
        {
          mapper.map(op->getOperand(i), mvinResults[i]);
        }
        Operation *cloned = op->clone(mapper);
        cloned->setLoc(op->getLoc());
        b.insert(cloned);

        // 添加mvoutop
        SmallVector<Value> mvoutResults;
        for (unsigned i = 0, e = cloned->getNumResults(); i < e; ++i)
        {
          auto result = cloned->getResult(i);
          auto mvoutOp = b.create<npux::mvoutOp>(op->getLoc(), result.getType(), result, "NPU_SPM");
          mvoutResults.push_back(mvoutOp.getResult());
        }

        // Return the results of the cloned op (may be zero or more)
        b.create<func::ReturnOp>(op->getLoc(), mvoutResults);

        // Replace original op with a call to the new function.
        OpBuilder callBuilder(op);
        auto call = callBuilder.create<func::CallOp>(op->getLoc(), outlined.getSymName(), retTypes, op->getOperands());
        if (!retTypes.empty())
          op->replaceAllUsesWith(call.getResults());

        // Emit logs
        op->emitRemark() << "outlined to @" << fnName;
        LLVM_DEBUG(llvm::dbgs() << "[OutlineNpuxOps] Outlined op '"
                                << op->getName() << "' to @" << fnName
                                << " (args=" << argTypes.size()
                                << ", rets=" << retTypes.size() << ")\n");
        ++totalOutlined;

        op->erase();
      }
    }

    module->emitRemark() << "OutlineNpuxOps: done (outlined="
                         << totalOutlined << ")";
    LLVM_DEBUG(llvm::dbgs() << "[OutlineNpuxOps] Done. Total outlined: "
                            << totalOutlined << "\n");
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createOutlineNpuxOpsPass() {
  return std::make_unique<OutlineNpuxOpsPass>();
}

void registerNpuxTransformPasses() {
  PassRegistration<OutlineNpuxOpsPass>();
}

} // namespace npux
