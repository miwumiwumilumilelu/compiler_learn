#include "LoopPasses.h"
#include "../utils/Matcher.h"
#include "AnalysisPasses.h"

using namespace sys;

// S-Expression
static Rule forCond("(lt (load x) y)");
static Rule forCondLe("(le (load x) y)");
// i = i + step
static Rule constIncr("(store (add (load x) y) x)");

std::map<std::string, int> RaiseToFor::stats() {
    return {
        { "raised-for-loops", raised },
    };
}

void RaiseToFor::run() {
    ArrayBase(module).run();
    Builder builder;

    auto loops = module->findAll<WhileOp>();
    for (auto loop : loops) {
        if (loop->atFront())
            continue;

        auto before = loop->getRegion(0);
        auto after = loop->getRegion(1);
        auto proceed = before->getLastBlock()->getLastOp();
        if (!proceed) continue;
        auto cond = proceed->DEF();

        Op* ivAddr = nullptr;
        Op* stop = nullptr;
        Op* stopvar = nullptr;

        // i < n
        if (forCond.match(cond)) {
            ivAddr = forCond.extract("x");
            stop = stopvar = forCond.extract("y");
        }

        // i <= n
        if (!ivAddr && forCondLe.match(cond)) {
            ivAddr = forCondLe.extract("x");
            stopvar = forCondLe.extract("y");

            // i <= n -> i < n + 1
            builder.setAfterOp(stopvar);
            auto one = builder.create<IntOp>({ new IntAttr(1) });
            stop = builder.create<AddIOp>({ stopvar, one });
        }

        if (!ivAddr)
            continue;

        // Stop Check
        // Check if stopvar is CallOp or Impure. If so, we cannot raise to for.
        if (isa<CallOp>(stopvar) || stopvar->has<ImpureAttr>())
            continue;

        // Check if stopvar is modified inside the loop.If so, we cannot raise to for.
        if (isa<LoadOp>(stopvar)) {
            auto addr = stopvar->DEF(0);
            bool bad = false;
            for (auto use : addr->getUses()) {
                if (isa<StoreOp>(use) && use->inside(loop)) {
                    bad = true;
                    break;
                }
            }
            if (bad) continue;
        }

        // Analysis Increment
        bool good = true;
        bool foundIncr = false;
        Op *incr = nullptr;

        for (auto use : ivAddr->getUses()) {
            if (!use->inside(loop) || isa<LoadOp>(use)) continue;

            if (!constIncr.match(use, {{"x", ivAddr}})) {
                good = false;
                break;
            }

            Op* vi = constIncr.extract("y");
            if (!foundIncr) {
                incr = vi;
                foundIncr = true;
            } 
            // Check if all increments are the same for if-else.
            else if (incr != vi) {
                if (!(isa<IntOp>(incr) && isa<IntOp>(vi) && V(incr) == V(vi))) {
                    good = false;
                    break;
                }
            }

            // Step size update must be the last operation of a basic block,
            // or the last step before a jump (Break/Continue).
            if(use->atBack()) {
                continue;
            }

            auto next = use->nextOp();
            if (isa<BreakOp>(next) || isa<ContinueOp>(next)) {
                continue;
            }

            good = false;
            break;
        }

        if (!good || !foundIncr) continue;

        // Dynamic Step safety check.
        // Incr is not constant.
        // BaseAttr Solve the alias problem.
        if (!isa<IntOp>(incr) && incr->inside(loop)) {
            if (isa<LoadOp>(incr)) {
                auto addr = incr->DEF();
                if (!addr->has<BaseAttr>() || SIZE(BASE(addr)) != 4)
                {
                    continue;
                }
                auto base = BASE(addr);
                auto stores = loop->findAll<StoreOp>();
                good = true;
                for (auto store : stores) {
                    auto saddr = store->DEF(1);
                    if (!saddr->has<BaseAttr>() || BASE(saddr) == base)
                    {
                        good = false;
                        break;
                    }
                }
                if (!good) continue;
            } else {
                continue;
            }
        }

        auto terms = loop->findAll<BreakOp>();
        auto conts = loop->findAll<ContinueOp>();
        std::copy(conts.begin(), conts.end(), std::back_inserter(terms));

        for (auto x : terms) {
            if (x->atFront() || !constIncr.match(x->prevOp())) {
                good = false;
                break;
            }
        }

        auto back = after->getLastBlock()->getLastOp();
        if (!good || !constIncr.match(back)) continue;

        // Rewrite Init
        Op *runner, *init;
        bool removable = true;
        for (runner = loop->prevOp(); !runner->atFront(); runner = runner->prevOp()) {
            // while/if/for
            if (runner->getRegionCount()) {
                auto stores = runner->findAll<StoreOp>();
                for (auto store : stores) {
                    if (store->DEF(1) == ivAddr) {
                        good = false;
                        break;
                    }
                }
                if (isa<ForOp>(runner) && ivAddr == runner->DEF(3)) {
                    init = runner->DEF(1); // ForOp stop
                    removable = false;
                    break;
                }
                continue;
            }

            // i = init
            if (isa<StoreOp>(runner) && runner->DEF(1) == ivAddr) {
                init = runner->DEF(0);
                break;
            }
            
            if (ivAddr->getUses().count(runner)) {
                removable = false;
            }
        }
        
        if (!good || !init || runner->atFront()) continue;

        // Rewriting

        // LICM
        if (isa<IntOp>(incr) && incr->inside(loop))
            incr->moveBefore(loop);

        if (removable)
            runner->erase();

        builder.setBeforeOp(loop);

        auto floop = builder.create<ForOp>({ init, stop, incr, ivAddr });
        auto region = floop->appendRegion();
        
        const auto &bbs = after->getBlocks();
        for (auto it = bbs.begin(); it != bbs.end();) {
            auto next = it; next++;
            (*it)->moveToEnd(region);
            it = next;
        }

        auto bb = before->getFirstBlock();
        bb->getLastOp()->erase(); // Remove ProceedOp.
        bb->inlineBefore(floop);

        std::vector<Op*> remove_list;
        for (auto use : ivAddr->getUses()) {
            if (!use->inside(floop)) {
                continue;
            }
            if (isa<StoreOp>(use)) {
                remove_list.push_back(use);
            } else if (isa<LoadOp>(use)) {
                use->replaceAllUsesWith(floop);
                remove_list.push_back(use);
            }
        }

        for (auto x : remove_list)
            x->erase();

        loop->erase();
        raised++;
    }
}
