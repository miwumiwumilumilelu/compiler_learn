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
    // lambda has Ret operator()(A a) const
    template<typename F,typename Ret, typename A>
    static A helper(Ret (F::*)(A) const);

    template<class F>
    using argument_t = decltype(helper(&F::operator()));

protected:
    ModuleOp *module; // The module to be optimized

    template<class F>
    void runRewriter(Op *op, F rewriter) {
        using T = std::remove_pointer_t<argument_t<F>>;
        bool success;
        int total = 0;
        do {
            if (++total > 10000) 
                break;
            
            auto ts = op->findAll<T>();
            success = false;
            for (auto t : ts) {
                success |= rewriter(cast<T>(t));
            }
        } while (success);
    }

    template<class F>
    void runRewriter(F rewriter) {
        runRewriter(module, rewriter);
    }

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