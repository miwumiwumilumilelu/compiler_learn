#include "LowerPasses.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"
#include <vector>
#include <set>

using namespace sys;

// IfOp: Break down if-then-else into basic block jumps.
static void handleIf(Op *x) {
    Builder builder;

    auto bb = x->getParent();
    auto region = bb->getParent();

    // Let the Op after IfOp become a new inserted block separately.
    auto beforeIf = bb;
    bb = region->insertAfter(bb);
    beforeIf->splitOpsAfter(bb, x);

    // Move then-region blocks after beforeIf.
    auto thenRegion = x->getRegion(0);
    auto [thenFirst, thenFinal] = thenRegion->moveTo(beforeIf);
    auto final =thenFinal;

    // (Optional) Move else-region blocks after thenFinal.
    bool hasElse = x->getRegions().size() > 1;
    if (hasElse) {
        auto elseRegion = x->getRegion(1);
        auto [elseFirst, elseFinal] = elseRegion->moveTo(thenFinal);
        final = elseFinal;
        
        builder.setToBlockEnd(beforeIf);
        builder.create<BranchOp>({ x->getOperand() }, {
            new TargetAttr(thenFirst),
            new ElseAttr(elseFirst)
        });
    }

    // The final block of thenRegion must connect to a "end" block.
    auto end = region->insertAfter(final);

    builder.setToBlockEnd(final);
    builder.create<GotoOp>({ new TargetAttr(end) });

    if (hasElse) {
        builder.setToBlockEnd(thenFinal);
        builder.create<GotoOp>({ new TargetAttr(end) });
    } else {
        builder.setToBlockEnd(beforeIf);
        builder.create<BranchOp>({ x->getOperand() }, {
            new TargetAttr(thenFirst),
            new ElseAttr(end)
        });
    }

    builder.setToBlockEnd(end);
    builder.create<GotoOp>({ new TargetAttr(bb) });
    x->erase(); // Remove the original IfOp.
}

static void handleWhile(Op *x) {
    Builder builder;

    auto bb = x->getParent();
    auto region = bb->getParent();

    auto beforeWhile = bb;
    bb = region->insertAfter(bb);
    beforeWhile->splitOpsAfter(bb, x);

    auto beforeRegion = x->getRegion(0);
    auto [beforeFirst, beforeFinal] = beforeRegion->moveTo(beforeWhile);

    auto afterRegion = x->getRegion(1);
    auto [afterFirst, afterFinal] = afterRegion->moveTo(beforeFinal);

    auto end = region->insertAfter(afterFinal);

    auto op = cast<ProceedOp>(beforeFinal->getLastOp());
    Value condition = op->getOperand();
    builder.setBeforeOp(op);
    builder.create<BranchOp>({ condition }, {
        new TargetAttr(afterFirst),
        new ElseAttr(end)
    });
    op->erase();

    // Loop back.
    builder.setToBlockEnd(afterFinal);
    builder.create<GotoOp>({ new TargetAttr(beforeFirst) });
    
    // Use unusedBB to deal with break/continue dead code.
    auto unusedBB = region->insertAfter(end);
    for (auto curr = beforeFirst; curr != end; curr = curr->nextBlock()) {
        std::vector<Op*> disrupters;
        for (auto op : curr->getOps()) {
            if (isa<BreakOp>(op) || isa<ContinueOp>(op) || isa<ReturnOp>(op)) {
                disrupters.push_back(op);
            }
        }

        for (auto op : disrupters) {
            // Consider dead code situations.
            auto parentBB = op->getParent();
            parentBB->splitOpsAfter(unusedBB, op);
            
            if (isa<BreakOp>(op)) {
                builder.setBeforeOp(op);
                builder.create<GotoOp>({ new TargetAttr(end) });
                op->erase();
            } else if (isa<ContinueOp>(op)) {
                builder.setBeforeOp(op);
                builder.create<GotoOp>({ new TargetAttr(beforeFirst) });
                op->erase();
            }

            for (auto garbage : unusedBB->getOps())
                garbage->erase();
        }
    }
    unusedBB->erase();

    builder.setToBlockEnd(end);
    builder.create<GotoOp>({ new TargetAttr(bb) });
    x->erase();
}

static bool isTerminator(Op *op) {
    return isa<GotoOp>(op) || isa<BranchOp>(op) || isa<ReturnOp>(op);
}

void tidy(FuncOp *func) {
    Builder builder;
    auto body = func->getRegion();
    auto last = body->getLastBlock();
    
    if (last->getOpCount() == 0 || !isa<ReturnOp>(last->getLastOp())) {
        builder.setToBlockEnd(body->getLastBlock());
        builder.create<ReturnOp>();
    }

    for (auto bb : body->getBlocks()) {
        Op *term = nullptr;
        for (auto op : bb->getOps()) {
            if (isTerminator(op)) {
                term = op;
                break;
            }
        }

        if (!term || term == bb->getLastOp()) continue;

        std::vector<Op*> remove;
        for (auto op = term->nextOp(); op; op = op->nextOp()) {
            op->removeAllOperands();
            remove.push_back(op);
        }
        for (auto op : remove)
            op->erase();
    }

    // Connect to pass-through basic blocks.
    for (auto it = body->begin(); it != body->end(); ++it) {
        auto bb = *it;
        auto next = it ; ++next;
        if (next != body->end()) {
            if(bb->getOpCount() == 0 || !isTerminator(bb->getLastOp())) {
                builder.setToBlockEnd(bb);
                builder.create<GotoOp>({ new TargetAttr(*next) });
            }
        }
    }

    body->updatePreds();

    // Elimination of intermediate jump blocks.
    // If a block contains only one Goto, optimize it away.
    // 1. Establish inliner map.
    std::map<BasicBlock*, BasicBlock*> inliner;
    for (auto bb : body->getBlocks()) {
        if (bb->getOpCount() != 1 || !isa<GotoOp>(bb->getLastOp())) 
            continue;
        
        auto last = bb->getLastOp();
        auto target = last->get<TargetAttr>();
        if (target->bb != bb) {
            inliner[bb] = target->bb;
        }
    }

    // 2. Define "update" function to update.
    // Lambda to update targets according to inliner map.
    auto update = [&](BasicBlock *&from) {
        while (inliner.count(from)) {
            from = inliner[from];
        }
    };

    // 3. Update all branches/gotos.
    for (auto bb : body->getBlocks()) {
        auto last = bb->getLastOp();
        if (last->has<TargetAttr>()) {
            auto target = last->get<TargetAttr>();
            update(target->bb);
        }
        if (last->has<ElseAttr>()) {
            auto ifnot = last->get<ElseAttr>();
            update(ifnot->bb);
        }
    }

    body->updatePreds();

    for (auto [k, v] : inliner) {
        k->erase();
    }

    body->updatePreds();

    // Move all stack space allocation instructions into this new entry block.
    if (body->getFirstBlock()->preds.size() >= 1) {
        auto first = body->getFirstBlock();
        auto entry = body->insert(first);
        auto ops = first->getOps();
        for (auto op : ops) {
            if (isa<AllocaOp>(op)) {
                op->moveToEnd(entry);
            }
        }
        builder.setToBlockEnd(entry);
        builder.create<GotoOp>({ new TargetAttr(first) });
    }
}

void FlattenCFG::run() {
    auto ifs = module->findAll<IfOp>();
    for (auto x : ifs) {
        handleIf(x);
    }

    auto whiles = module->findAll<WhileOp>();
    for (auto x : whiles) {
        handleWhile(x);
    }

    auto funcs = collectFuncs();
    for (auto func : funcs) {
        tidy(func);
    }
}
