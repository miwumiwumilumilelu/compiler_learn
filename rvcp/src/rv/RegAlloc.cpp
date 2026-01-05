/// Priority-based Greedy Graph Coloring
#include "RvPasses.h"
#include "Regs.h"
#include <unordered_set>

using namespace sys;
using namespace sys::rv;

namespace
{
// Overflow flag attribute.
class SpilledRdAttr : public AttrImpl<SpilledRdAttr, RVLINE + 2097152> {
public:
    bool fp;
    int offset;
    Op* ref;

    SpilledRdAttr(bool fp, int offset, Op* ref) : fp(fp), offset(offset), ref(ref) {}

    std::string toString() override {
        return "<rd-spilled = " + std::to_string(offset) + (fp ? "f" : "") + ">";
    }
    SpilledRdAttr* clone() override { return new SpilledRdAttr(fp, offset, ref); }
};

class SpilledRsAttr : public AttrImpl<SpilledRsAttr, RVLINE + 2097152> {
public:
    bool fp;
    int offset;
    Op* ref;

    SpilledRsAttr(bool fp, int offset, Op* ref) : fp(fp), offset(offset), ref(ref) {}

    std::string toString() override {
        return "<rs-spilled = " + std::to_string(offset) + (fp ? "f" : "") + ">";
    }
    SpilledRsAttr* clone() override { return new SpilledRsAttr(fp, offset, ref); }
};

class SpilledRs2Attr : public AttrImpl<SpilledRs2Attr, RVLINE + 2097152> {
public:
    bool fp;
    int offset;
    Op* ref;

    SpilledRs2Attr(bool fp, int offset, Op* ref) : fp(fp), offset(offset), ref(ref) {}

    std::string toString() override {
        return "<rs2-spilled = " + std::to_string(offset) + (fp ? "f" : "") + ">";  
    }
    SpilledRs2Attr* clone() override { return new SpilledRs2Attr(fp, offset, ref); }
};
    
} // namespace

std::map<std::string, int> RegAlloc::stats() {
    return { 
        {"spilled", spilled}, 
        {"convertedTotal", convertedTotal}
    };
}

#define GET_SPILLED_ARGS(op) \
    (fpreg(op->getResultType()), spillOffset[op], op)

#define ADD_ATTR(Index, AttrTy) \
    auto v##Index = op->getOperand(Index).defining; \
    if (!spillOffset.count(v##Index)) \
        op->add<AttrTy>(getReg(v##Index)); \
    else \
        op->add<Spilled##AttrTy> GET_SPILLED_ARGS(v##Index);

#define BINARY ADD_ATTR(0, RsAttr); ADD_ATTR(1, Rs2Attr);
#define UNARY ADD_ATTR(0, RsAttr);

#define LOWER(Ty, Body) \
    runRewriter(funcOp, [&](Ty *op) \
    { \
        if (op->getOperands().size() == 0) \
            return false; \
        Body \
        op->removeAllOperands(); \
        return true; \
    });

#define CREATE_MV(fp, rd, rs) \
    if (!fp) \
        builder.create<MvOp>({ RDC(rd), RSC(rs) }); \
    else \
        builder.create<FmvOp>({ RDC(rd), RSC(rs) });

// Used in LiveOut.
struct Event {
    int timestamp;
    bool start;
    Op *op;
};

// Implemented in OpBase.cpp.
std::string getValueNumber(Value value);

void dumpInterf(Region *region, const std::unordered_map<Op*, std::set<Op*>> &interf) {
    region->dump(std::cerr, /*depth=*/1);
    std::cerr << "\n\n===== interference graph =====\n\n";
    for (auto [k, v] : interf) {
        std::cerr << getValueNumber(k->getResult()) << ": ";
        for (auto op : v)
            std::cerr << getValueNumber(op->getResult()) << " ";
        std::cerr << "\n";
  }
}

void dumpAssignment(Region *region, const std::unordered_map<Op*, Reg> &assignment) {
    region->dump(std::cerr, /*depth=*/1);
    std::cerr << "\n\n===== assignment =====\n\n";
    for (auto [k, v] : assignment) {
        std::cerr << getValueNumber(k->getResult()) << " = " << showReg(v) << "\n";
    }
}

void RegAlloc::runImpl(Region *region, bool isLeaf) {
    // Select the allocation order table based on function type.
    const Reg *order = isLeaf ? leafOrder : normalOrder;
    const Reg *orderf = isLeaf ? leafOrderf : normalOrderf;
    const int regcount = isLeaf ? leafRegCnt : normalRegCnt;
    const int regcountf = isLeaf ? leafRegCntf : normalRegCntf;

    Builder builder;

    // pre-Lowering (force replace some ops)
    auto funcOp = region->getParent();
    runRewriter(funcOp, [&](EqOp *op) {
        builder.setBeforeOp(op);
        // eq a, b  =>  xor t, a, b; seqz d, t
        auto xorOp = builder.create<XorOp>(op->getOperands(), op->getAttrs());
        builder.replace<SeqzOp>(op,{ xorOp });
        return true;
    });

    runRewriter(funcOp, [&](NeOp *op) {
        builder.setBeforeOp(op);
        // ne a, b  =>  xor t, a, b; snez d, t
        auto xorOp = builder.create<XorOp>(op->getOperands(), op->getAttrs());
        builder.replace<SnezOp>(op,{ xorOp });
        return true;
    });

    runRewriter(funcOp, [&](LeOp *op) {
        builder.setBeforeOp(op);
        auto l = op->getOperand(0);
        auto r = op->getOperand(1);
        // le a, b  =>  lt b, a (即 !(b < a)) => seqz (b < a)
        // Turn (l <= r) into !(r < l).
        auto xorOp = builder.create<SltOp>({ r, l }, op->getAttrs());
        builder.replace<SeqzOp>(op,{ xorOp });
        return true;
    });

    runRewriter(funcOp, [&](LtOp *op) {
        // lt a, b => slt a, b
        builder.replace<SltOp>(op, op->getOperands(), op->getAttrs());
        return true;
    });

    std::map<Op*, Reg> assignment;

    // Handle function calls (CallOp)
    // Purpose: To protect the Caller-Saved registers (t0-t6, a0-a7, etc.).
    // Principle: After the Call instruction is executed, the values ​​of these registers will change. 
    // If a variable survives beyond the Call instruction,
    // It cannot be assigned to the Caller-Saved register.
    // We artificially create "conflicts" by inserting PlaceHolderOp and forcibly allocating it to these registers.
    runRewriter(funcOp, [&](CallOp *op) {
        std::vector<Op*> writes;
        for (auto runner = op->prevOp(); runner && isa<WriteRegOp>(runner); runner = runner->prevOp())
            writes.push_back(runner);

        // To prevent reused
        for (int i = 0 ;i < int(writes.size()) - 1; i++) {
            builder.setBeforeOp(writes[i]);
            for (int j = i + 1; j< writes.size(); j++) {
                auto reg = REG(writes[j]);
                auto placeholder = builder.create<PlaceHolderOp>();
                assignment[placeholder] = reg; 
                if (isFP(reg)) placeholder->setResultType(Value::f32);
            }
        }

        builder.setBeforeOp(op);
        for (auto reg : callerSaved) {
            auto placeholder = builder.create<PlaceHolderOp>();
            assignment[placeholder] = reg;
            if (isFP(reg)) placeholder->setResultType(Value::f32);
        }
        return false;
    });

    // Process the function parameters (GetArgOp)
    // Objective: The function's parameters are fixed in a0-a7 (fa0-fa7).
    // We need to ensure that these registers are occupied at the function entry point.  
    builder.setToRegionStart(region);
    std::vector<Value> argHolders, fargsHolders;
    auto argcnt = funcOp->get<ArgCountAttr>()->count;
    for (int i = 0; i < std::min(argcnt, 8); i++) {
        auto placeholder = builder.create<PlaceHolderOp>();
        assignment[placeholder] = argRegs[i];
        argHolders.push_back(placeholder);

        auto fplaceholdr = builder.create<PlaceHolderOp>();
        assignment[fplaceholdr] = fargRegs[i];
        fargsHolders.push_back(fplaceholdr);
    }

    auto rawGets = funcOp->findAll<GetArgOp>();
    std::vector<Op*> getArgs;
    getArgs.resize(argcnt);
    // V(x) is index of x in args. .e.g 0 1 2
    for (auto x : rawGets) getArgs[V(x)] = x;

    int fcnt = 0, cnt = 0;
    BasicBlock *entry = region->getFirstBlock();
    
    for (size_t i = 0; i < getArgs.size(); i++) {
        // If a parameter is not used at all in the function, skip it.
        if (!getArgs[i]) continue;
        Op *op = getArgs[i];
        auto ty = op->getResultType();

        if (fpreg(ty) && fcnt < 8) {
            op->moveToStart(entry);
            builder.setBeforeOp(op);
            builder.create<PlaceHolderOp>({ fargsHolders[fcnt] });
            builder.replace<ReadRegOp>(op, Value::f32, {new RegAttr(fargRegs[fcnt])});
            fcnt++;
            continue;
        }

        if (!fpreg(ty) && cnt < 8) {
            op->moveToStart(entry);
            builder.setBeforeOp(op);
            builder.create<PlaceHolderOp>({ argHolders[cnt] });
            builder.replace<ReadRegOp>(op, Value::i32, {new RegAttr(argRegs[cnt])});
            cnt++;
            continue;
        }

        // TODO: spill
    }

    // std::cerr << "--- After Pre-coloring (Step 1) ---\n";
    // region->dump(std::cerr, 1);

    region->updateLiveness();

    std::unordered_map<Op*, std::set<Op*>> interf, spillInterf;
    std::unordered_map<Op*, int> priority;
    std::unordered_map<Op*, Op*> prefer;
    std::unordered_map<Op*, std::vector<Op*>> phiOperand;

    int currentPriority = 2;

    for (auto bb : region->getBlocks()) {
        // Use-Def chain
        // Reverse analysis.
        std::unordered_map<Op*, int> lastUsed, defined;
        const auto &ops = bb->getOps();
        auto it = ops.end();

        for (int i = (int) ops.size() - 1; i >= 0; i--) {
            auto op = *--it;

            // if (op->getName() == "rv.call") {
            //     std::cerr << "Found CallOp at index " << i << ", operands count: " << op->getOperands().size() << "\n";
            //     for(auto v : op->getOperands()) {
            //         std::cerr << "  Uses: " << getValueNumber(v) << "\n";
            //     }
            // }           

            for( auto v : op->getOperands()) {
                if (!lastUsed.count(v.defining))
                    lastUsed[v.defining] = i;
            }
            defined[op] = i;

            if (!lastUsed.count(op))
                lastUsed[op] = i + 1;

            if (isa<WriteRegOp>(op)) {
                assignment[op] = REG(op);
                priority[op] = 1; 
            }
            if(isa<ReadRegOp>(op))
                priority[op] = 1;

            // The immediate value range for RISC-V I-Type instructions is [-2048, 2047].
            if (isa<LiOp>(op) && (V(op) <= 2047 && V(op) >= -2048))
                priority[op] = -2;

            if (isa<PhiOp>(op)) {
                priority[op] = currentPriority + 1;
                for (auto x : op->getOperands()) {
                    priority[x.defining] = currentPriority;
                    prefer[x.defining] = op;
                    phiOperand[op].push_back(x.defining);
                }
                currentPriority += 2;
            }
        }

        for (auto op : bb->getLiveOut())
            lastUsed[op] = ops.size();

        std::vector<Event> events;
        for (auto [op, v] : lastUsed) {
            if (defined[op] == v) continue;

            events.push_back(Event { defined[op], true, op });
            events.push_back(Event { v, false, op });
        }

        std::sort(events.begin(), events.end(), [](Event a, Event b) {
            return a.timestamp == b.timestamp ? (!a.start && b.start) : a.timestamp < b.timestamp;
        });

        std::unordered_set<Op*> active;
        for (const auto& event : events) {
            auto op = event.op;
            if (isa<JOp>(op)) continue;

            if (event.start) {
                // Conflict
                for (Op* activeOp : active) {
                    if (fpreg(activeOp->getResultType()) ^ fpreg(op->getResultType())) {
                        // Stack Slot conflict.
                        spillInterf[activeOp].insert(op);
                        spillInterf[op].insert(activeOp);
                        continue;
                    }
                    interf[activeOp].insert(op);
                    interf[op].insert(activeOp);
                }
                active.insert(op);
            } else {
                active.erase(op);
            }
        }
    }

    // std::cerr << "--- Interference Graph (Step 2) ---\n";
    // dumpInterf(region, interf);

    std::vector<Op*> ops;
    for (auto [k, v] : interf) {
        ops.push_back(k);
    }
    for (auto [k, v] : priority) {
        ops.push_back(k);
    }

    // 1. Allocation is done first for higher priority (pa > pb).
    // 2. If the priorities are the same, then the allocation is done for the node with larger degree (interf.size() > interf.size()).
    // The more difficult the variable to assign, the earlier it should be processed; the remaining easy variables can be filled in later.
    std::sort(ops.begin(), ops.end(), [&](Op* a, Op* b) {
        auto pa = priority[a];
        auto pb = priority[b];
        return pa == pb ? interf[a].size() > interf[b].size() : pa > pb;
    });

    std::unordered_map<Op*, int> spillOffset;
    int currentOffset = STACKOFF(funcOp);
    int highest = 0;

    for (auto op : ops) {
        if(assignment.count(op))
            continue;

        std::unordered_set<Reg> bad, unpreferred;

        // sp (stack pointer) and zero are always considered non-conflicting/read-only and are not counted as being occupied.
        for (auto v : interf[op]) {
            if (assignment.count(v) && assignment[v] != Reg::sp && assignment[v] != Reg::zero)
                bad.insert(assignment[v]);
        }

        // Phi avoid registers of conflicting objects for all their operands.
        if (isa<PhiOp>(op)) {
            const auto &operands = phiOperand[op];
            for (auto x : operands) {
                for (auto v : interf[x]) {
                    if (assignment.count(v) && assignment[v] != Reg::sp && assignment[v] != Reg::zero)
                        unpreferred.insert(assignment[v]);
                }
            }
        }

        // Phi operands
        if (prefer.count(op)) {
            auto ref = prefer[op];
            if (assignment.count(ref) && !bad.count(assignment[ref])) {
                assignment[op] = assignment[ref];
                continue;
            }
        }
        
        // If one of the Uses of Op is WriteRegOp (precolored write), try to align.
        int preferred = -1;
        for (auto use : op->getUses()) {
            if (isa<WriteRegOp>(use)) {
                auto reg = REG(use);
                if (!bad.count(reg)) {
                    preferred = (int) reg;
                    break;
                }
            }
        }
        if (isa<ReadRegOp>(op)) {
            auto reg = REG(op);
            if (!bad.count(reg)) {
                preferred = (int) reg;
            }
        }
        if (preferred != -1) {
            assignment[op] = (Reg) preferred;
            continue;
        }

        // Try to allocate a register.
        auto rcnt = !fpreg(op->getResultType()) ? regcount : regcountf;
        auto rorder = !fpreg(op->getResultType()) ? order : orderf;

        for (int i = 0; i < rcnt; i++) {
            if (!bad.count(rorder[i]) && !unpreferred.count(rorder[i])) {
                assignment[op] = rorder[i];
                break;
            }
        }

        if (!assignment.count(op) && unpreferred.size()) {
            for (int i = 0; i < rcnt; i++) {
                if (!bad.count(rorder[i])) {
                    assignment[op] = rorder[i];
                    break;
                }
            }
        }

        if (assignment.count(op)) {
            continue;
        }

        // Spill.
        spilled++;

        int desired = currentOffset;
        std::unordered_set<int> conflict;

        for (auto v : interf[op]) {
            if (!spillOffset.count(v)) continue;
            conflict.insert(spillOffset[v]);
        }

        for (auto v : spillInterf[op]) {
            if (!spillOffset.count(v)) continue;
            conflict.insert(spillOffset[v]);
        }

        while (conflict.count(desired)) 
            desired += 8;

        spillOffset[op] = desired;
        
        highest = std::max(highest, desired);
    }

    if (highest == currentOffset) {
        // Only one overflow, just borrow s10/fs10.
        for (auto [op, _] : spillOffset) 
            assignment[op] = fpreg(op->getResultType()) ? fspillReg : spillReg;
        spillOffset.clear();
    }

    if (highest == currentOffset + 8) {
        // Only two overflow, just borrow s10/fs10 and s11/fs11.
        for (auto [op, offset] : spillOffset) {
            auto fp = fpreg(op->getResultType());
            assignment[op] = (offset > currentOffset) ? (fp ? fspillReg2 : spillReg2) : (fp ? fspillReg : spillReg);
        }
        spillOffset.clear();
    }

    // FP Reuse for Spilling.
    if (spillOffset.size()) {
        std::unordered_set<Reg> used;
        for (auto [op, x] : assignment) {
            if (isa<PlaceHolderOp>(op)) continue;
            used.insert(x);
        }

        std::unordered_map<int, Reg> fpmv;
        auto off = STACKOFF(funcOp);
        for (auto reg : leafOrderf) {
            if (highest <= off)
                break; 
            if (used.count(reg) || (!isLeaf && calleeSaved.count(reg))) 
                continue;

            fpmv[highest] = reg;
            highest -= 8;
        }

        // Map the offset to a negative value.
        // And use this to make subsequent judgments; change it to fmv.
        for (auto &[_, offset] : spillOffset) {
            if (fpmv.count(offset))
                offset = -int(fpmv[offset]);
        }
    }

    if (spillOffset.size()) 
        STACKOFF(funcOp) = highest + 8;
    
    // Returns default registers to prevent crashes.
    const auto getReg = 
    [&](Op* op) {
        return assignment.count(op) ? assignment[op] :
            fpreg(op->getResultType()) ? orderf[0] : order[0];
    };

    LOWER(AddOp, BINARY);
    LOWER(AddwOp, BINARY);
    LOWER(SubOp, BINARY);
    LOWER(SubwOp, BINARY);
    LOWER(MulOp, BINARY);
    LOWER(MulwOp, BINARY);
    LOWER(MulhOp, BINARY);
    LOWER(MulhuOp, BINARY);
    LOWER(DivwOp, BINARY);
    LOWER(DivOp, BINARY);
    LOWER(RemOp, BINARY);
    LOWER(RemwOp, BINARY);

    LOWER(BneOp, BINARY);
    LOWER(BeqOp, BINARY);
    LOWER(BltOp, BINARY);
    LOWER(BgeOp, BINARY);

    LOWER(AndOp, BINARY);
    LOWER(OrOp, BINARY);
    LOWER(XorOp, BINARY);

    LOWER(SltOp, BINARY);
    LOWER(SllwOp, BINARY);
    LOWER(SrlwOp, BINARY);
    LOWER(SrawOp, BINARY);
    LOWER(SllOp, BINARY);
    LOWER(SrlOp, BINARY);
    LOWER(SraOp, BINARY);

    LOWER(FaddOp, BINARY);
    LOWER(FsubOp, BINARY);
    LOWER(FmulOp, BINARY);
    LOWER(FdivOp, BINARY);
    LOWER(FeqOp, BINARY);
    LOWER(FltOp, BINARY);
    LOWER(FleOp, BINARY);

    LOWER(StoreOp, BINARY);

    LOWER(LoadOp, UNARY);
    LOWER(AddiwOp, UNARY);
    LOWER(AddiOp, UNARY);
    LOWER(SlliwOp, UNARY);
    LOWER(SrliwOp, UNARY);
    LOWER(SraiwOp, UNARY);
    LOWER(SraiOp, UNARY);
    LOWER(SlliOp, UNARY);
    LOWER(SrliOp, UNARY);
    LOWER(SeqzOp, UNARY);
    LOWER(SnezOp, UNARY);
    LOWER(SltiOp, UNARY);
    LOWER(AndiOp, UNARY);
    LOWER(OriOp, UNARY);
    LOWER(XoriOp, UNARY);

    LOWER(FcvtswOp, UNARY);
    LOWER(FcvtwsRtzOp, UNARY);
    LOWER(FmvwxOp, UNARY);


    // Cleanup Operands.
    // The operands of CallOp, RetOp, and PlaceHolderOp are only used to establish conflict relationships during allocation.
    // After allocation, these operands are no longer needed.
    for (auto bb : region->getBlocks()) {
        for (auto op : bb->getOps()) {
            if (isa<PlaceHolderOp>(op) || isa<CallOp>(op) || isa<RetOp>(op)) 
                op->removeAllOperands();
        }
    }

    // Remove placeholders.
    auto holders = funcOp->findAll<PlaceHolderOp>();
    for (auto holder : holders) 
        holder->erase();

    runRewriter(funcOp, [&](WriteRegOp *op) {
        builder.setBeforeOp(op);
        CREATE_MV(isFP(REG(op)), REG(op), getReg(op->DEF(0)));
        auto mv = op->prevOp();

        if (spillOffset.count(op->DEF(0))) {
            mv->remove<RsAttr>();
            mv->add<SpilledRsAttr> GET_SPILLED_ARGS(op->DEF(0));    
        }

        op->erase();
        return false;
    });

    runRewriter(funcOp, [&](ReadRegOp *op) {
        builder.setBeforeOp(op);
        CREATE_MV(isFP(REG(op)), getReg(op), REG(op));
        auto mv = op->prevOp();

        assignment[mv] = getReg(op);
        if (spillOffset.count(op)) {
            mv->remove<RdAttr>();
            mv->add<SpilledRdAttr> GET_SPILLED_ARGS(op);    
            spillOffset[mv] = spillOffset[op];
        }

        op->replaceAllUsesWith(mv);
        op->erase();
        return false;
    });

    std::vector<Op*> allPhis;
    auto bbs = region->getBlocks();
    // Split Critical Edges by inserting new Basic Blocks.
    for (auto bb : bbs) {
        if (bb->succs.size() <= 1) continue;
        
        auto edge1 = region->insertAfter(bb);
        auto edge2 = region->insertAfter(bb);
        auto bbTerm = bb->getLastOp();

        auto target = bbTerm->get<TargetAttr>();
        auto oldTarget = target->bb;
        target->bb = edge1;
        builder.setToBlockEnd(edge1);
        builder.create<JOp>({ new TargetAttr(oldTarget) });

        auto ifnot = bbTerm->get<ElseAttr>();
        auto oldElse = ifnot->bb;
        ifnot->bb = edge2;
        builder.setToBlockEnd(edge2);
        builder.create<JOp>({ new TargetAttr(oldElse) });

        for (auto succ : bb->succs) {
            for (auto phis : succ->getPhis()) {
                for (auto attr : phis->getAttrs()) {
                    auto from = cast<FromAttr>(attr);
                    if (from->bb != bb) continue;
                    if (succ == oldTarget) {
                        from->bb = edge1;
                    } 
                    if (succ == oldElse) {
                        from->bb = edge2;
                    }
                }
            }
        }
    }

#define SOFFSET(op, Ty) ((Reg)(-(op)->get<Spilled##Ty##Attr>()->offset - 1000))
#define SPILLABLE(op, Ty) (op->has<Ty##Attr>() ? op->get<Ty##Attr>()->reg : SOFFSET(op, Ty))

    std::unordered_map<BasicBlock*, std::vector<std::pair<Reg, Reg>>> moveMap;
    std::unordered_map<BasicBlock*, std::map<std::pair<Reg, Reg>, Op*>> revMap;

    for (auto bb : bbs) {
        auto phis = bb->getPhis();
        std::vector<Op*> moves;

        for (auto phi : phis) {
            auto &ops = phi->getOperands();
            auto &attrs = phi->getAttrs();

            for (size_t i = 0; i < ops.size(); i++) {
                auto fromBB = FROM(attrs[i]);
                auto term = fromBB->getLastOp();
                builder.setBeforeOp(term);
                auto def = ops[i].defining;

                Op *mv;
                bool isFp = fpreg(phi->getResultType());
                if (isFp) {
                    mv = builder.create<FmvOp>({
                        new ImpureAttr,
                        spillOffset.count(phi) ? (Attr*) new SpilledRdAttr GET_SPILLED_ARGS(phi) : RDC(getReg(phi)),
                        spillOffset.count(def) ? (Attr*) new SpilledRsAttr GET_SPILLED_ARGS(def) : RSC(getReg(def))
                    });
                } else {
                    mv = builder.create<MvOp>({
                        new ImpureAttr,
                        spillOffset.count(phi) ? (Attr*) new SpilledRdAttr GET_SPILLED_ARGS(phi) : RDC(getReg(phi)),
                        spillOffset.count(def) ? (Attr*) new SpilledRsAttr GET_SPILLED_ARGS(def) : RSC(getReg(def))
                    });
                }
                moves.push_back(mv);
            }
        }

        std::copy(phis.begin(), phis.end(), std::back_inserter(allPhis));

        for (auto mv : moves) {
            auto dst = SPILLABLE(mv, Rd);
            auto src = SPILLABLE(mv, Rs);
            if (dst == src) {
                mv->erase();
                continue;
            }

            auto parent = mv->getParent();
            moveMap[parent].emplace_back(dst, src);
            revMap[parent][{dst, src}] = mv;
        }
    }

    for (const auto &[bb, mvs] : moveMap) {
        std::unordered_map<Reg, Reg> moveGraph;
        for (auto [dst, src] : mvs) {
            moveGraph[dst] = src;
        }

        std::set<Reg> visited, visiting;
        std::vector<std::pair<Reg, Reg>> sorted;
        std::vector<Reg> headers;
        std::unordered_map<Reg, std::vector<Reg>> members;
        std::unordered_set<Reg> inCycle;

        // DFS
        std::function<void(Reg)> dfs = [&](Reg node) {
            visiting.insert(node);
            Reg src = moveGraph[node];
            if (visiting.count(src)) {
                // A node is visited twice. Here's a cycle.
                headers.push_back(node);
            } else if (!visited.count(src) && moveGraph.count(src)) {
                dfs(src); // No access and dependent on others.
            }
            visiting.erase(node);
            visited.insert(node);
            sorted.emplace_back(node, src);
        };

        for (auto [dst, src] : mvs) {
            if (!visited.count(dst)) {
                dfs(dst);
            }
        }

        std::reverse(sorted.begin(), sorted.end());

        for (auto header : headers) {
            Reg cur = header;
            do {
                members[header].push_back(cur);
                cur = moveGraph[cur];
            } while (cur != header);

            for (auto member : members[header]) {
                inCycle.insert(member);
            }
        }

        Op *term = bb->getLastOp();

        std::unordered_set<Reg> emitted;
        for (auto [dst, src] : sorted) {
            if (dst == src || emitted.count(dst) || inCycle.count(dst)) 
                continue;

            revMap[bb][{dst, src}]->moveBefore(term);
            emitted.insert(dst);
        }

        if (members.empty()) 
            continue;

        // Back up the value of the loop header to a temporary register (s11/fspillReg2).
        for (auto header : headers) {
            const auto &cycle = members[header];
            assert(!cycle.empty());

            Reg headerSrc = moveGraph[header];
            auto mv = revMap[bb][{ header, headerSrc }];
            bool fp = isFP(header);
            Reg tmp = fp ? fspillReg2 : spillReg2;

            RD(mv) = tmp;
            mv->moveBefore(term);

            Reg curr = headerSrc;
            while (curr != header) {
                Reg nextSrc = moveGraph[curr];
                revMap[bb][{ curr, nextSrc }]->moveBefore(term);
                curr = nextSrc;
            }

            builder.setBeforeOp(term);
            CREATE_MV(fp, header, tmp);
        }
    }
    
    for (auto phi : allPhis) {
        phi->removeAllOperands();
    }
    for (auto phi : allPhis) {
        // Debugging aid: if there are still uses, dump the module.
        if (phi->getUses().size())
          module->dump();
        phi->erase();
    }

    for (auto bb : region->getBlocks()) {
        for (auto op : bb->getOps()) {
            if (hasRd(op) && !op->has<RdAttr>() && !op->has<SpilledRdAttr>()) {
                if (!spillOffset.count(op))
                    op->add<RdAttr>(getReg(op));
                else
                    op->add<SpilledRdAttr> GET_SPILLED_ARGS(op);
            }
        }
    }
    
    // Deal with spill variables.
    std::vector<Op*> remove;
    for (auto bb : region->getBlocks()) {
        int delta = 0;
        // Track Stack Pointer
        for (auto op : bb->getOps()) {
            if (isa<SubSpOp>(op)) {
                delta += V(op);
                continue;
            }
            // Spilled Write -> Store.
            if (auto rd = op->find<SpilledRdAttr>()) {
                if(isa<LiOp>(rd->ref) || isa<LaOp>(rd->ref)) {
                    remove.push_back(op);
                    continue;
                }

                int offset = delta + rd->offset;
                bool fp = rd->fp;

                auto reg = fp ? fspillReg : spillReg;
                
                builder.setAfterOp(op);

                if (offset < delta)
                    builder.create<FmvdxOp>({ RDC(Reg(delta - offset)), RSC(reg) });
                else if (offset < 2048) // [-2048, 2047]
                    // store reg, offset(sp)
                    builder.create<StoreOp>({ 
                        RSC(reg),
                        RS2C(Reg::sp),
                        new IntAttr(offset),
                        new SizeAttr(8)
                    });
                else if (offset < 4096) {
                    // addi spillReg2, sp, 2047
                    builder.create<AddiOp>({
                        RDC(spillReg2),
                        RSC(Reg::sp),
                        new IntAttr(2047),
                        new SizeAttr(8)
                    });
                    // store reg, (offset - 2047)(spillReg2)
                    builder.create<StoreOp>({
                        RSC(reg),
                        RSC(spillReg2),
                        new IntAttr(offset - 2047),
                        new SizeAttr(8)
                    });
                }
                else assert(false);

                op->add<RdAttr>(reg);
            }

            // Spilled Read -> Load.
            if (auto rs = op->find<SpilledRsAttr>()) {
                int offset = delta + rs->offset;
                bool fp = rs->fp;
                auto reg = fp ? fspillReg : spillReg;
                // lw/ld/flw/fld
                auto ldty = fp? Value::f32 : Value::i64;

                builder.setBeforeOp(op);

                auto ref = rs->ref;
                if (isa<LiOp>(ref))
                    builder.create<LiOp>({ RDC(reg), new IntAttr(V(ref)) });
                else if (isa<LaOp>(ref))
                    builder.create<LaOp>({ RDC(reg), new NameAttr(NAME(ref)) });
                else if (offset < delta)
                    // fmvxd reg, spillReg
                    builder.create<FmvxdOp>({ RDC(reg), RSC(Reg(delta - offset)) });
                else if (offset < 2048) // [-2048, 2047]
                    // load reg, offset(sp)
                    builder.create<LoadOp>(ldty, {
                        RDC(reg),
                        RSC(Reg::sp),
                        new IntAttr(offset),
                        new SizeAttr(8)
                    });
                else if (offset < 4096) {
                    // addi spillReg, sp, 2047
                    builder.create<AddiOp>({
                        RDC(spillReg),
                        RSC(Reg::sp),
                        new IntAttr(2047),
                        new SizeAttr(8)
                    });
                    // load reg, (offset - 2047)(spillReg)
                    builder.create<LoadOp>(ldty, {
                        RDC(reg),
                        RSC(spillReg),
                        new IntAttr(offset - 2047),
                        new SizeAttr(8)
                    });
                }
                else assert(false);

                op->add<RsAttr>(reg);
            }
            
            if (auto rs2 = op->find<SpilledRs2Attr>()) {
                int offset = delta + rs2->offset;
                bool fp = rs2->fp;
                auto reg = fp ? fspillReg2 : spillReg2;
                // lw/ld/flw/fld
                auto ldty = fp? Value::f32 : Value::i64;

                builder.setBeforeOp(op);

                auto ref = rs2->ref;
                if (isa<LiOp>(ref))
                    builder.create<LiOp>({ RDC(reg), new IntAttr(V(ref)) });
                else if (isa<LaOp>(ref))
                    builder.create<LaOp>({ RDC(reg), new NameAttr(NAME(ref)) });
                else if (offset < delta)
                    // fmvxd reg, spillReg
                    builder.create<FmvxdOp>({ RDC(reg), RSC(Reg(delta - offset)) });
                else if (offset < 2048) // [-2048, 2047]
                    // load reg, offset(sp)
                    builder.create<LoadOp>(ldty, {
                        RDC(reg),
                        RSC(Reg::sp),
                        new IntAttr(offset),
                        new SizeAttr(8)
                    });
                else if (offset < 4096) {
                    // addi spillReg, sp, 2047
                    builder.create<AddiOp>({
                        RDC(spillReg2),
                        RSC(Reg::sp),
                        new IntAttr(2047),
                        new SizeAttr(8)
                    });
                    // load reg, (offset - 2047)(spillReg)
                    builder.create<LoadOp>(ldty, {
                        RDC(reg),
                        RSC(spillReg2),
                        new IntAttr(offset - 2047),
                        new SizeAttr(8)
                    });
                }
                else assert(false);

                op->add<Rs2Attr>(reg);
            }
        }
    }

    for (auto op : remove)
        op->erase();

}



void RegAlloc::run() {
    auto funcs = collectFuncs();
    std::set<FuncOp*> leaves;
    for (auto func : funcs) {
        auto calls = func->findAll<sys::rv::CallOp>();
        if (calls.size() == 0) {
            leaves.insert(func);
        }
        runImpl(func->getRegion(), calls.size() == 0);
    }

    for (auto func : funcs) {
        // For proEpilogue.
        auto &set = usedRegisters[func];
        for (auto bb : func->getRegion()->getBlocks()) {
            for (auto op : bb->getOps()) {
                if (op->has<RdAttr>()) 
                    set.insert(op->get<RdAttr>()->reg);
                if (op->has<RsAttr>()) 
                    set.insert(op->get<RsAttr>()->reg);
                if (op->has<Rs2Attr>()) 
                    set.insert(op->get<Rs2Attr>()->reg);
            }
        }
    }

    for (auto func : funcs) {
        proEpilogue(func, leaves.count(func));
        tidyup(func->getRegion());
    }

}



