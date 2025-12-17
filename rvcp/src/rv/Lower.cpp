#include "RvPasses.h"
#include "RvOps.h"
#include "RvAttrs.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

using namespace sys::rv;
using namespace sys;

static void rewriteAlloca(FuncOp* func) {
    Builder builder;
    auto region = func->getRegion();
    auto block = region->getFirstBlock();

    size_t offset = 0;
    size_t total = 0;
    std::vector<AllocaOp*> allocas;

    for (auto op : block->getOps()) {
        if (auto allocaOp = dyn_cast<AllocaOp>(op)) {
            total += SIZE(op);
            allocas.push_back(allocaOp);
        }
    }

    for (auto op : allocas) {
        builder.setBeforeOp(op);
        auto spValue = builder.create<ReadRegOp>(Value::i32, { new RegAttr(Reg::sp) });
        auto offsetValue = builder.create<LiOp>({ new IntAttr(offset) });
        auto add = builder.create<AddOp>({ spValue, offsetValue });
        op->replaceAllUsesWith(add);

        offset += SIZE(op);
        op->erase();
    }

    func->add<StackOffsetAttr>(total);
}

#define REPLACE(BeforeTy, AfterTy) \
    runRewriter([&](BeforeTy* op) { \
        builder.replace<AfterTy>(op, op->getOperands(), op->getAttrs()); \
        return true; \
    });

void Lower::run() {
    Builder builder;
// lambda is used to rewrite all ops.
// runRewriter( [&](decltype(op) op){} );
    runRewriter([&](PhiOp* op) {
        if(op->getResultType() == Value::f32) return false;
        for (auto operand : op->getOperands()) {
            if (operand.defining->getResultType() == Value::f32) {
                op->setResultType(Value::f32);
                return true;
            }
        }
        return false;
    });

    runRewriter([&](SelectOp* op) {
        auto x = op->DEF(0), y = op->DEF(1), z = op->DEF(2);
        auto parent = op->getParent();
        auto region = parent->getParent();

        auto tgt = region->appendBlock();
        auto bb1 = region->appendBlock();
        auto bb2 = region->appendBlock();

        parent->splitOpsAfter(tgt, op);
        tgt->moveAfter(parent);
        bb1->moveBefore(tgt);
        bb2->moveBefore(tgt);

        builder.setToBlockEnd(parent);
        builder.create<BranchOp>({ x }, { new TargetAttr(bb1), new ElseAttr(bb2) });

        builder.setToBlockEnd(bb1);
        builder.create<GotoOp>({ new TargetAttr(tgt) });
        builder.setToBlockEnd(bb2);
        builder.create<GotoOp>({ new TargetAttr(tgt) });

        builder.replace<PhiOp>(op, { y, z }, { new FromAttr(bb1), new FromAttr(bb2) });

        for (auto succ : parent->succs) {
        for (auto phi : succ->getPhis()) {
            for (auto attr : phi->getAttrs()) {
            if (FROM(attr) == parent) FROM(attr) = tgt;
            }
        }
        succ->preds.erase(parent);
        succ->preds.insert(tgt);
        }
        tgt->succs = parent->succs;
        parent->succs = { bb1, bb2 };
        return false;
    });

    REPLACE(IntOp, LiOp);
    REPLACE(AddIOp, AddwOp); REPLACE(AddLOp, AddOp);
    REPLACE(SubIOp, SubwOp); REPLACE(SubLOp, SubOp);
    REPLACE(MulIOp, MulwOp); REPLACE(MulLOp, MulOp);
    REPLACE(MulshOp, MulhOp); REPLACE(MuluhOp, MulhuOp);
    REPLACE(DivIOp, DivwOp); REPLACE(DivLOp, DivOp);
    REPLACE(ModIOp, RemwOp); REPLACE(ModLOp, RemOp);
    REPLACE(LShiftOp, SllwOp); REPLACE(LShiftLOp, SllOp);
    REPLACE(RShiftOp, SrawOp); REPLACE(RShiftLOp, SraOp);
    REPLACE(GotoOp, JOp);
    REPLACE(GetGlobalOp, LaOp);
    REPLACE(AndIOp, AndOp); REPLACE(OrIOp, OrOp); REPLACE(XorIOp, XorOp);
    REPLACE(AddFOp, FaddOp); REPLACE(SubFOp, FsubOp);
    REPLACE(MulFOp, FmulOp); REPLACE(DivFOp, FdivOp);
    REPLACE(LtFOp, FltOp);   REPLACE(EqFOp, FeqOp);   REPLACE(LeFOp, FleOp);
    REPLACE(F2IOp, FcvtwsRtzOp); REPLACE(I2FOp, FcvtswOp);    

    runRewriter([&](FloatOp *op) {
        float value = F(op);
        builder.setBeforeOp(op);
        auto li = builder.create<LiOp>({ new IntAttr(*(int*) &value) });
        builder.replace<FmvwxOp>(op, { li });
        return true;
    });

    runRewriter([&](sys::LoadOp *op) {
        auto load = builder.replace<sys::rv::LoadOp>(op, op->getResultType(), op->getOperands(), op->getAttrs());
        load->add<IntAttr>(0);
        return true;
    });
    runRewriter([&](sys::StoreOp *op) {
        auto store = builder.replace<sys::rv::StoreOp>(op, op->getOperands(), op->getAttrs());
        store->add<IntAttr>(0);
        return true;
    });

    runRewriter([&](BranchOp *op) {
        auto cond = op->getOperand().defining;
        if (isa<EqOp>(cond)) { builder.replace<BeqOp>(op, cond->getOperands(), op->getAttrs()); return true; }
        if (isa<NeOp>(cond)) { builder.replace<BneOp>(op, cond->getOperands(), op->getAttrs()); return true; }
        if (isa<LeOp>(cond)) { builder.replace<BgeOp>(op, { cond->getOperand(1), cond->getOperand(0) }, op->getAttrs()); return true; }
        if (isa<LtOp>(cond)) { builder.replace<BltOp>(op, cond->getOperands(), op->getAttrs()); return true; }

        builder.setBeforeOp(op);
        auto zero = builder.create<ReadRegOp>(Value::i32, { new RegAttr(Reg::zero) });
        builder.replace<BneOp>(op, { cond, zero }, op->getAttrs());
        return true;
    });

    REPLACE(LtOp, SltOp);

    // Para regs
    const static Reg regs[] = {
        Reg::a0, Reg::a1, Reg::a2, Reg::a3, Reg::a4, Reg::a5, Reg::a6, Reg::a7
    };
    const static Reg fregs[] = {
        Reg::fa0, Reg::fa1, Reg::fa2, Reg::fa3, Reg::fa4, Reg::fa5, Reg::fa6, Reg::fa7
    };

    runRewriter([&](CallOp *op) {
        builder.setBeforeOp(op);
        const auto &args = op->getOperands();
        // Spilled : spill args to stack.
        std::vector<Value> argsNew, fargsNew, spilled;

        for (size_t i = 0; i < args.size(); i++) {
            Value arg = args[i];
            if (arg.defining->getResultType() == Value::f32 && fargsNew.size() < 8) {
                fargsNew.push_back(builder.create<WriteRegOp>({ arg }, {new RegAttr(fregs[fargsNew.size()])}));
            }
            else if (arg.defining->getResultType() != Value::f32 && argsNew.size() < 8) {
                argsNew.push_back(builder.create<WriteRegOp>({ arg }, {new RegAttr(regs[argsNew.size()])}));
            }
            else {
                spilled.push_back(arg);
            }
        }
        // 8 bits
        int stackOffset = spilled.size() * 8;
        if (stackOffset % 16 != 0) stackOffset = (stackOffset / 16 + 1) * 16;

        if (stackOffset) 
            builder.create<SubSpOp>({ new IntAttr(stackOffset) });
        for (int i = 0; i < spilled.size(); i++) {
            auto sp = builder.create<ReadRegOp>(Value::i32, {new RegAttr(Reg::sp)});
            builder.create<StoreOp>({spilled[i], sp }, { new SizeAttr(8), new IntAttr(i * 8) });
        }

        builder.create<sys::rv::CallOp>(argsNew, { op->get<NameAttr>(), new ArgCountAttr(args.size()) });

        if (stackOffset > 0) builder.create<SubSpOp>({ new IntAttr(-stackOffset) });

        if (op->getResultType() == Value::f32)
        builder.replace<ReadRegOp>(op, Value::f32, { new RegAttr(Reg::fa0) });
        else
        builder.replace<ReadRegOp>(op, Value::i32, { new RegAttr(Reg::a0) });
        return true;
    });
    
    runRewriter([&](ReturnOp *op) {
        builder.setBeforeOp(op);
        if (op->getOperands().size()) {
            auto isFloat = op->DEF(0)->getResultType() == Value::f32;
            auto retReg = builder.create<WriteRegOp>({ op->getOperand() }, { new RegAttr(isFloat ? Reg::fa0 : Reg::a0) });
            builder.replace<RetOp>(op, { retReg });
        } else {
            builder.replace<RetOp>(op);
        }
        return true;
    });

    auto funcs = collectFuncs();
    for (auto func : funcs) rewriteAlloca(func);
}