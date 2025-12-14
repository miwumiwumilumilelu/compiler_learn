#ifndef PASS_H
#define PASS_H

#include <string>
#include <map>
#include <vector>
#include "../codegen/Ops.h"

namespace sys
{

    using DomTree = std::unordered_map<BasicBlock *, std::vector<BasicBlock *>>;

    class Pass
    {
    protected:
        ModuleOp *module; // The module to be optimized
        std::vector<FuncOp *> collectFuncs();
        DomTree getDomTree(Region *region);

    public:
        Pass(ModuleOp *module) : module(module) {}
        virtual ~Pass() {}

        //
        virtual std::string name() = 0;
        virtual std::map<std::string, int> stats() = 0;
        virtual void run() = 0;
    };

}

#endif // PASS_H