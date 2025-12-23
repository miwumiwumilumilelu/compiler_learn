#include "AnalysisPasses.h"
#include "../codegen/CodeGen.h"

using namespace sys;

namespace {

void remove(Region *region) {
    for (auto bb : region->getBlocks()) {
        for (auto op : bb->getOps()) {
            op->remove<BaseAttr>();
            for (auto r : op->getRegions())
                remove(r);
        }
    }
}

}

void ArrayBase::runImpl(Region *region) {
    for (auto bb : region->getBlocks()) {
        for (auto op : bb->getOps()) {
            for (auto r : op->getRegions())
                runImpl(r);
        
            if (isa<AllocaOp>(op) || isa<GetGlobalOp>(op) || isa<GetArgOp>(op)) {
                op->add<BaseAttr>(op);
                continue;
            }
            
            if (isa<AddLOp>(op)) {
                auto x = op->DEF(0);
                auto y = op->DEF(1);
                
                if (!x->has<BaseAttr>()) {
                    if (y->has<BaseAttr>())
                        std::swap(x, y);
                    else 
                        continue;
                }

                op->add<BaseAttr>(BASE(x));
                continue;
            }
        }
    }
}

void ArrayBase::run() {
    auto funcs = collectFuncs();
    
    for (auto func : funcs) {
        Region *region = func->getRegion();
        // Remove all op <BaseAttr>.
        remove(region);

        auto bb = region->getFirstBlock();
        
        if (bb->getOpCount() && isa<AllocaOp>(bb->getFirstOp()))
            bb = bb->nextBlock();

        // Put Global in the first basic block.
        Builder builder;
        auto gets = func->findAll<GetGlobalOp>();
        std::unordered_map<std::string, Op*> hoisted;

        for (auto get : gets) {
            const auto &name = NAME(get);
            if (!hoisted.count(name)) {
                builder.setToBlockStart(bb);
                auto newget = builder.create<GetGlobalOp>({ new NameAttr(name) });
                hoisted[name] = newget;
            }
            get->replaceAllUsesWith(hoisted[name]);
            get->erase();
        }
        // At this point, the analysis will be very efficient 
        // because the base addresses of all global variables 
        // are converged into a single set of Ops.
        runImpl(region);
    }
}