#ifndef CLEANUP_PASSES_H
#define CLEANUP_PASSES_H

#include "pass.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

namespace sys {

// Dead code elimination. Deals with functions, basic blocks and variables.
class DCE : public Pass {
    std::vector<Op*> removeable;
    int elimOp = 0;
    int elimFn = 0;
    int elimBB = 0;
    bool elimBlocks;

    bool isImpure(Op *op);
    bool markImpure(Region *region);
    void runOnRegion(Region *region);

    std::map<std::string, FuncOp*> fnMap;
public:
    // If DCE is called before flatten cfg, then it shouldn't eliminate blocks,
    // since the blocks aren't actually well-formed.
    DCE(ModuleOp *module, bool elimBlocks = true): Pass(module), elimBlocks(elimBlocks) {}
        
    std::string name() override { return "dce"; };
    std::map<std::string, int> stats() override;
    void run() override;
};

}