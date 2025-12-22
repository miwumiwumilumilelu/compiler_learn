#ifndef ANALYSISPASSES_H
#define ANALYSISPASSES_H

#include "../CFG-opt/pass.h"
#include "../codegen/Ops.h"
#include "../codegen/Attrs.h"
#include "PreAttrs.h"

namespace sys {

// Marks base of an array.
class ArrayBase : public Pass {
    void runImpl(Region *region);
public:
    ArrayBase(ModuleOp *module): Pass(module) {}
        
    std::string name() override { return "array-base"; };
    std::map<std::string, int> stats() override { return {}; };
    void run() override;
};

}

#endif // ANALYSISPASSES_H