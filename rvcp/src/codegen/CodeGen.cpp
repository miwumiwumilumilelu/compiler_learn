#include "CodeGen.h"
#include "../utils/DynamicCast.h"
#include "Attrs.h"
#include "OpBase.h"
#include "Ops.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace sys;

void Builder::setToRegionStart(Region *region) {
    setToBlockStart(region->getFirstBlock());
}

void Builder::setToRegionEnd(Region *region) {
    setToBlockEnd(region->getFirstBlock());
}

void Builder::setToBlockStart(BasicBlock *block) {
    bb = block;
    at = bb->begin();
    init = true;
}

void Builder::setToBlockEnd(BasicBlock *block) {
    bb = block;
    at = bb->end();
    init = true;
}

void Builder::setBeforeOp(Op *op) {
    bb = op->parent;
    at = op->place;
    init = true;
}

void Builder::setAfterOp(Op *op) {
    setBeforeOp(op);
    ++at;
}

// shallow-copies operands, deep-copies attrs.
Op *Builder::copy(Op *op) {
    auto opnew = new Op(op->opid, op->resultTy, op->operands);
    for (auto attr : op->attrs) {
        auto cloned = attr->clone();
        cloned->refcnt++;
        opnew->attrs.push_back(cloned);
    }
    opnew->opname = op->opname;
    bb->insert(at, opnew);
    return opnew;
}

CodeGen::CodeGen(ASTNode *node): module(new ModuleOp()) {
    module->createFirstBlock();
    builder.setToRegionStart(module->getRegion());
    emit(node);
}

int CodeGen::getSize(Type *ty) {
    assert(ty);
    if (isa<IntType>(ty) || isa<FloatType>(ty))
        return 4;
    if (auto arrTy = dyn_cast<ArrayType>(ty))
        return getSize(arrTy->base) * arrTy->getSize();

    return 8;
}

Value CodeGen::emitBinary (BinaryNode *node) {
//   enum {
//     Add, Sub, Mul, Div, Mod, And, Or,
//     // >= and > Canonicalized.
//     Eq, Ne, Le, Lt
//   } kind;

    if (node->kind == BinaryNode::And) {
        auto alloca = builder.create<AllocaOp>({ new SizeAttr(4) });
        //   l && r
        // becomes
        //   if (l)
        //     %1 = not_zero r
        //     store %1, %alloca
        //   else
        //     store 0, %alloca
        //   load %alloca
        auto l = emitExpr(node->l);
        auto branch = builder.create<IfOp>({ l });
        {
            auto ifso = branch->appendRegion();
            auto block = ifso->appendBlock();
            Builder::Guard guard(builder);

            builder.setToBlockStart(block);
            auto r = emitExpr(node->r);
            auto snez = builder.create<SetNotZeroOp>({ r });
            builder.create<StoreOp>({ snez, alloca }, { new SizeAttr(4) });
        }
        {
            auto ifnot = branch->appendRegion();
            auto block = ifnot->appendBlock();
            Builder::Guard guard(builder);

            builder.setToBlockStart(block);
            auto zero = builder.create<IntOp>({ new IntAttr(0) });
            // implicit zero because of Value(Op op*)
            builder.create<StoreOp>({ zero, alloca }, { new SizeAttr(4) });
        }
        return builder.create<LoadOp>(Value::i32, { alloca }, { new SizeAttr(4) });
    }
    
    if (node->kind == BinaryNode::Or) {
        auto alloca = builder.create<AllocaOp>({ new SizeAttr(4) });
        //   l || r
        // becomes
        //   if (l)
        //     store 1, %alloca
        //   else
        //     %1 = not_zero r
        //     store %1, %alloca
        //   load %alloca
        auto l = emitExpr(node->l);
        auto branch = builder.create<IfOp>({ l });
        {
            auto ifso = branch->appendRegion();
            auto block = ifso->appendBlock();
            Builder::Guard guard(builder);

            builder.setToBlockStart(block);
            auto one = builder.create<IntOp>({ new IntAttr(1) });
            builder.create<StoreOp>({ one, alloca }, { new SizeAttr(4) });
        }
        {
            auto ifnot = branch->appendRegion();
            auto block = ifnot->appendBlock();
            Builder::Guard guard(builder);

            builder.setToBlockStart(block);
            auto r = emitExpr(node->r);
            auto snez = builder.create<SetNotZeroOp>({ r });
            builder.create<StoreOp>({ snez, alloca }, { new SizeAttr(4) });
        }
    }

    auto l = emitExpr(node->l);
    auto r = emitExpr(node->r);
    if (!isa<FloatType>(node->l->type) && !isa<FloatType>(node->r->type)) {
        switch (node->kind) {
            case BinaryNode::Add:
                return builder.create<AddIOp>({ l, r });
            case BinaryNode::Sub:
                return builder.create<SubIOp>({ l, r });
            case BinaryNode::Mul:
                return builder.create<MulIOp>({ l, r });
            case BinaryNode::Div:
                return builder.create<DivIOp>({ l, r });
            case BinaryNode::Mod:
                return builder.create<ModIOp>({ l, r });
            case BinaryNode::Eq:
                return builder.create<EqOp>({ l, r });
            case BinaryNode::Ne:
                return builder.create<NeOp>({ l, r });
            case BinaryNode::Lt:
                return builder.create<LtOp>({ l, r });
            case BinaryNode::Le:
                return builder.create<LeOp>({ l, r });
            default:
                assert(false);
        }
    } else {
        switch (node->kind) {
            case BinaryNode::Add:
                return builder.create<AddFOp>({ l, r });
            case BinaryNode::Sub:
                return builder.create<SubFOp>({ l, r });
            case BinaryNode::Mul:
                return builder.create<MulFOp>({ l, r });
            case BinaryNode::Div:
                return builder.create<DivFOp>({ l, r });
            case BinaryNode::Mod:
                return builder.create<ModFOp>({ l, r });
            case BinaryNode::Eq:
                return builder.create<EqFOp>({ l, r });
            case BinaryNode::Ne:
                return builder.create<NeFOp>({ l, r });
            case BinaryNode::Lt:
                return builder.create<LtFOp>({ l, r });
            case BinaryNode::Le:
                return builder.create<LeFOp>({ l, r });
            default:
                std::cerr << "unsupported float binary " << node->kind << "\n";
                assert(false);
        }
    }
}

Value CodeGen::emitExpr(ASTNode *node) {

}

Value CodeGen::emitUnary(UnaryNode *node) {
    
}

void CodeGen::emit(ASTNode *node) {

}