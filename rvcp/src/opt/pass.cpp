#include "pass.h"

using namespace sys;

std::vector<FuncOp*> Pass::collectFuncs() {
    std::vector<FuncOp*> result;

    auto region = module->getRegion();
    if (!region) return result;

    auto block = region->getFirstBlock();
    if (!block) return result;

    for (auto op : block->getOps()) {
        if (auto func = dyn_cast<FuncOp>(op)) {
            result.push_back(func);
        }
    }

    return result;
}
