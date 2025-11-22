#ifndef OPBASE_H
#define OPBASE_H

#include "../utils/DynamicCast.h"
#include <list>
#include <string>

namespace sys {

class Op;
class BasicBlock;

class Value {
public:
    Op *defining;
    enum Type {
        unit, i32, i64, f32, i128, f128
    };
    Value(){};
    Value(Op *frome);

    bool operator==(Value x) const { return defining == x.defining; }
    bool operator!=(Value x) const { return defining != x.defining; }
    bool operator<(Value x) const { return defining < x.defining; }
    bool operator>(Value x) const { return defining > x.defining; }
    bool operator<=(Value x) const { return defining <= x.defining; }
    bool operator>=(Value x) const { return defining >= x.defining; }
};

// CFG (control flow graph)
class Region {
    std::list<BasicBlock*> bb;
    Op *parent;
    void showLiveIn();
public:
    using iterator = decltype(bb)::iterator;

    auto &getBlocks() { return bb; }
    BasicBlock *getFirstBlock() { return *bb.begin(); }
    BasicBlock *getLastBlock() { return *--bb.end(); }

    iterator begin() { return bb.begin(); }
    iterator end() { return bb.end(); }

    Op *getParent() { return parent; }

    BasicBlock *appendBlock();
    void dump(std::ostream &os, int depth);

    BasicBlock *insert(BasicBlock* at);
    BasicBlock *insertAfter(BasicBlock* at);
    void remove(BasicBlock* at);

    void insert(iterator at, BasicBlock *bb);
    void insertAfter(iterator at, BasicBlock *bb);
    void remove(iterator at);

    void updatePreds();
    void updateDoms();
    void updateDomFront();
    void updatePDoms();
    void updateLiveness();
    std::pair<BasicBlock*, BasicBlock*> moveTo(BasicBlock *insertionPoint);

    void erase();
    Region(Op *parent): parent(parent) {}
};

}

#endif // OPBASE_H