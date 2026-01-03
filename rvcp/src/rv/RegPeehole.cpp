#include "RvPasses.h"
#include "Regs.h"

using namespace sys::rv;
using namespace sys;

// 辅助宏：快速创建 Move 指令
#define CREATE_MV(fp, rd, rs) \
    if (!fp) \
        builder.create<MvOp>({ RDC(rd), RSC(rs) }); \
    else \
        builder.create<FmvOp>({ RDC(rd), RSC(rs) });

// ==========================================
// 1. 晚期窥孔优化 (Late Peephole)
// ==========================================
// 负责在分配寄存器后，合并 Store/Load，消除冗余 Move

// 分支替换宏：用于规范化分支指令
#define REPLACE_BRANCH(T1, T2) \
    REPLACE_BRANCH_IMPL(T1, T2); \
    REPLACE_BRANCH_IMPL(T2, T1)

#define GENERATE_J \
    builder.setAfterOp(op); \
    builder.create<JOp>({ new TargetAttr(ifnot) })

#define END_REPLACE \
    op->remove<ElseAttr>(); \
    return true

#define REPLACE_BRANCH_IMPL(BeforeTy, AfterTy) \
    runRewriter(funcOp, [&](BeforeTy *op) { \
        if (!op->has<ElseAttr>()) return false; \
        auto &target = TARGET(op); \
        auto ifnot = ELSE(op); \
        auto me = op->getParent(); \
        if (me == me->getParent()->getLastBlock()) { \
            GENERATE_J; END_REPLACE; \
        } \
        if (me->nextBlock() == target) { \
            builder.replace<AfterTy>(op, { op->get<RsAttr>(), op->get<Rs2Attr>(), new TargetAttr(ifnot), }); \
            return true; \
        } \
        if (me->nextBlock() == ifnot) return false; \
        GENERATE_J; END_REPLACE; \
    })

int RegAlloc::latePeephole(Op *funcOp) {
    Builder builder;
    int converted = 0;

    // 优化: Store 后紧跟 Load (且地址相同) -> Load 换成 Move
    runRewriter(funcOp, [&](StoreOp *op) {
        if (op->atBack()) return false;
        auto next = op->nextOp();
        if (isa<LoadOp>(next) && RS(next) == RS2(op) && V(next) == V(op) && SIZE(next) == SIZE(op)) {
            converted++;
            builder.setBeforeOp(next);
            CREATE_MV(isFP(RD(next)), RD(next), RS(op));
            next->erase();
            return false;
        }
        return false;
    });

    // 优化: Fsd 后紧跟 Fld (浮点存取优化)
    runRewriter(funcOp, [&](FmvdxOp *op) {
        if (op->atBack()) return false;
        auto next = op->nextOp();
        if (isa<FmvxdOp>(next) && RS(next) == RD(op)) {
            converted++;
            builder.setBeforeOp(next);
            CREATE_MV(isFP(RD(next)), RD(next), RS(op));
            next->erase();
            return false;
        }
        return false;
    });

    // 优化: 合并连续的 Store (Store Folding)
    bool changed;
    do {
        changed = false;
        auto stores = funcOp->findAll<StoreOp>();
        for (auto op : stores) {
            if (op == op->getParent()->getLastOp()) continue;
            auto next = op->nextOp();
            // sw zero, N(sp); sw zero, N+4(sp) -> sd zero, N(sp)
            if (isa<StoreOp>(next) && RS(op) == Reg::zero && RS2(op) == Reg::sp &&
                RS(next) == Reg::zero && RS2(next) == Reg::sp &&
                SIZE(op) == 4 && SIZE(next) == 4) {

                if (V(op) % 8 == 0 && V(next) == V(op) + 4) {
                    converted++;
                    changed = true;
                    builder.replace<StoreOp>(op, { RSC(Reg::zero), RS2C(Reg::sp), new IntAttr(V(op)), new SizeAttr(8) });
                    next->erase();
                    break;
                }
                if (V(op) % 8 == 4 && V(next) == V(op) - 4) {
                    converted++;
                    changed = true;
                    builder.replace<StoreOp>(op, { RSC(Reg::zero), RS2C(Reg::sp), new IntAttr(V(op) - 4), new SizeAttr(8) });
                    next->erase();
                    break;
                }
            }
        }
    } while (changed);

    // 优化: 消除无用的 Move
    runRewriter(funcOp, [&](MvOp *op) {
        if (RD(op) == RS(op)) {
            converted++;
            op->erase();
            return true;
        }
        if (!op->atBack()) {
            auto mv2 = op->nextOp();
            if (isa<MvOp>(mv2) && RD(op) == RS(mv2) && RS(op) == RD(mv2)) {
                op->erase();
                std::swap(RD(mv2), RS(mv2)); // 消除对称 Move
            }
        }
        return false;
    });

    runRewriter(funcOp, [&](FmvOp *op) {
        if (RD(op) == RS(op)) {
            converted++;
            op->erase();
            return true;
        }
        return false;
    });

    return converted;
}

// ==========================================
// 2. 清理工作 (Tidy Up)
// ==========================================
// 负责清理跳板、合并基本块、规范化分支

void RegAlloc::tidyup(Region *region) {
    Builder builder;
    auto funcOp = region->getParent();
    region->updatePreds();

    // 1. 消除单跳转块 (Jump Threading)
    std::map<BasicBlock*, BasicBlock*> jumpTo;
    for (auto bb : region->getBlocks()) {
        if (bb->getOpCount() == 1 && isa<JOp>(bb->getLastOp())) {
            jumpTo[bb] = bb->getLastOp()->get<TargetAttr>()->bb;
        }
    }
    bool changed;
    do {
        changed = false;
        for (auto [k, v] : jumpTo)
            if (jumpTo.count(v)) {
                jumpTo[k] = jumpTo[v];
                changed = true;
            }
    } while (changed);

    for (auto bb : region->getBlocks()) {
        auto term = bb->getLastOp();
        if (auto target = term->find<TargetAttr>())
            if (jumpTo.count(target->bb)) target->bb = jumpTo[target->bb];
        if (auto ifnot = term->find<ElseAttr>())
            if (jumpTo.count(ifnot->bb)) ifnot->bb = jumpTo[ifnot->bb];
    }
    region->updatePreds();
    for (auto [bb, v] : jumpTo) bb->erase();

    // 2. 基本块合并
    do {
        changed = false;
        const auto &bbs = region->getBlocks();
        for (auto bb : bbs) {
            if (bb->succs.size() != 1) continue;
            auto succ = *bb->succs.begin();
            if (succ->preds.size() != 1) continue;

            if (isa<JOp>(bb->getLastOp())) bb->getLastOp()->erase();

            for (auto s : succ->succs) {
                s->preds.erase(succ);
                s->preds.insert(bb);
                bb->succs.insert(s);
            }
            bb->succs.erase(succ);
            auto ops = succ->getOps();
            for (auto op : ops) op->moveToEnd(bb);
            succ->forceErase();
            changed = true;
            break;
        }
    } while (changed);

    // 3. 规范化分支
    REPLACE_BRANCH(BltOp, BgeOp);
    REPLACE_BRANCH(BeqOp, BneOp);
    REPLACE_BRANCH(BleOp, BgtOp);

    // 4. 执行晚期窥孔
    int converted;
    do {
        converted = latePeephole(funcOp);
        convertedTotal += converted;
    } while (converted);

    // 5. 消除无用 Jump
    runRewriter(funcOp, [&](JOp *op) {
        auto me = op->getParent();
        if (me != me->getParent()->getLastBlock() && me->nextBlock() == TARGET(op)) {
            op->erase();
            return true;
        }
        return false;
    });
}

// ==========================================
// 3. 序言与结语生成 (Prologue & Epilogue)
// ==========================================

#define CREATE_STORE(addr, offset) \
    if (isFP(reg)) \
        builder.create<FsdOp>({ RSC(reg), RS2C(addr), new IntAttr(offset) }); \
    else \
        builder.create<StoreOp>({ RSC(reg), RS2C(addr), new IntAttr(offset), new SizeAttr(8) });

void save(Builder builder, const std::vector<Reg> &regs, int offset) {
    using sys::rv::StoreOp;
    for (auto reg : regs) {
        offset -= 8;
        if (offset < 2048) {
            CREATE_STORE(Reg::sp, offset)
        } else {
            builder.create<LiOp>({ RDC(spillReg2), new IntAttr(offset) });
            builder.create<AddOp>({ RDC(spillReg2), RSC(spillReg2), RS2C(Reg::sp) });
            CREATE_STORE(spillReg2, 0);
        }
    }
}

#define CREATE_LOAD(addr, offset) \
    if (isFP(reg)) \
        builder.create<FldOp>({ RDC(reg), RSC(addr), new IntAttr(offset) }); \
    else \
        builder.create<LoadOp>(ty, { RDC(reg), RSC(addr), new IntAttr(offset), new SizeAttr(8) });

void load(Builder builder, const std::vector<Reg> &regs, int offset) {
    using sys::rv::LoadOp;
    for (auto reg : regs) {
        offset -= 8;
        auto isFloat = isFP(reg);
        Value::Type ty = isFloat ? Value::f32 : Value::i64;
        if (offset < 2048) {
            CREATE_LOAD(Reg::sp, offset)
        } else {
            builder.create<LiOp>({ RDC(spillReg), new IntAttr(offset) });
            builder.create<AddOp>({ RDC(spillReg), RSC(spillReg), RS2C(Reg::sp) });
            CREATE_LOAD(spillReg, 0);
        }
    }
}

void RegAlloc::proEpilogue(FuncOp *funcOp, bool isLeaf) {
    Builder builder;
    auto usedRegs = usedRegisters[funcOp];
    auto region = funcOp->getRegion();

    // 1. 确定需要保存的寄存器
    std::vector<Reg> preserve;
    for (auto x : usedRegs)
        if (calleeSaved.count(x)) preserve.push_back(x);
    if (!isLeaf) preserve.push_back(Reg::ra);

    // 2. 计算栈大小
    int &offset = STACKOFF(funcOp);
    offset += 8 * preserve.size();
    if (offset % 16 != 0) offset = offset / 16 * 16 + 16;

    // 3. 生成序言
    auto entry = region->getFirstBlock();
    builder.setToBlockStart(entry);
    if (offset != 0) builder.create<SubSpOp>({ new IntAttr(offset) });
    save(builder, preserve, offset);

    // 4. 生成结语 (统一返回块)
    if (offset != 0) {
        auto rets = funcOp->findAll<RetOp>();
        auto bb = region->appendBlock();
        for (auto ret : rets) builder.replace<JOp>(ret, { new TargetAttr(bb) });

        builder.setToBlockStart(bb);
        load(builder, preserve, offset);
        builder.create<SubSpOp>({ new IntAttr(-offset) });
        builder.create<RetOp>();
    }

    // 5. 处理栈传参 (修正偏移)
    auto remainingGets = funcOp->findAll<GetArgOp>();
    std::sort(remainingGets.begin(), remainingGets.end(), [](Op *a, Op *b) { return V(a) < V(b); });
    std::map<Op*, int> argOffsets;
    int argOffset = 0;
    for (auto op : remainingGets) {
        argOffsets[op] = argOffset;
        argOffset += 8;
    }

    runRewriter(funcOp, [&](GetArgOp *op) {
        int myoffset = offset + argOffsets[op];
        builder.setBeforeOp(op);
        builder.replace<LoadOp>(op, isFP(RD(op)) ? Value::f32 : Value::i64, {
            RDC(RD(op)), RSC(Reg::sp), new IntAttr(myoffset), new SizeAttr(8)
        });
        return false;
    });

    // 6. 替换 SubSpOp 为真实指令
    runRewriter(funcOp, [&](SubSpOp *op) {
        int val = V(op);
        if (val <= 2048 && val > -2048) {
            builder.replace<AddiOp>(op, { RDC(Reg::sp), RSC(Reg::sp), new IntAttr(-val) });
        } else {
            builder.setBeforeOp(op);
            builder.create<LiOp>({ RDC(Reg::t0), new IntAttr(val) });
            builder.replace<SubOp>(op, { RDC(Reg::sp), RSC(Reg::sp), RS2C(Reg::t0) });
        }
        return true;
    });
}