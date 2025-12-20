#ifndef HIRPASSES_H
#define HIRPASSES_H

#include "../CFG-opt/pass.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

namespace sys {

class MoveAlloca : public Pass {
public:
    MoveAlloca(ModuleOp *module) : Pass(module) {}

    std::string name() override { return "move-alloca"; };
    std::map<std::string, int> stats() override { return {}; };

    void run() override;
};

// Lower operations back to its original form.
class ForOpLower : public Pass {
public:
    ForOpLower(ModuleOp *module): Pass(module) {}
    
    std::string name() override { return "forop-lower"; }
    std::map<std::string, int> stats() override { return {}; }
    void run() override;
};

}

#endif // HIRPASSES_H