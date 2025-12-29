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
    std::map<Op*, Reg> assignment;

    auto funcOp = region->getParent();

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
}

void RegAlloc::run() {
    auto funcs = collectFuncs();
    for (auto func : funcs) {
        bool isLeaf = (func->findAll<sys::rv::CallOp>().size() == 0);
        runImpl(func->getRegion(), isLeaf);
    }
}



