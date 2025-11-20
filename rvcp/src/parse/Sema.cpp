#include "Sema.h"
#include "ASTNode.h"
#include "../utils/DynamicCast.h"
#include "Type.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <set>

using namespace sys;

// decay array type to pointer type
PointerType* Sema::decay(ArrayType* arrTy) {
    std::vector<int> dims;
    for (int i = 1; i < arrTy->dims.size(); ++i) {
        dims.push_back(arrTy->dims[i]);
    }
    if (!dims.size()) {  
        return ctx.create<PointerType>(arrTy -> base);
    }
    return ctx.create<PointerType>(ctx.create<ArrayType>(arrTy->base, dims));
}

// raise pointer type to array type
ArrayType* Sema::raise(PointerType* ptr) {
    std::vector<int> dims {1};
    Type *base;
    if (auto pointee = dynamic_cast<ArrayType*>(ptr->baseType)) {
        for (auto x : pointee->dims) {
            dims.push_back(x);
        }
        base = pointee->base;
    }
    else {
        base = ptr->baseType;
    }
    return ctx.create<ArrayType>(base, dims);
}

Type *Sema::infer(ASTNode *node) {
    
    if (auto fn = dyn_cast<FnDeclNode>(node)) {
        assert(fn->type);
        auto fnTy = cast<FunctionType>(fn->type);
        symbols[fn->name] = fnTy;
        currentFunc = fnTy;

        SemanticScope scope(*this);
        for (int i = 0; i < fn->params.size(); ++i) {
            symbols[fn->params[i]] = fnTy->params[i];
        }

        for(auto x : fn->body->nodes) {
            infer(x);
        }
        return ctx.create<VoidType>();
    }

    if (auto blk = dyn_cast<BlockNode>(node)) {
        SemanticScope scope(*this);
        for (auto x : blk->nodes) {
            infer(x);
        }
        return ctx.create<VoidType>();
    }

    if (auto blk = dyn_cast<TransparentBlockNode>(node)) {
        for (auto x : blk->nodes) {
            infer(x);
        }
        return ctx.create<VoidType>();
    }

    if (isa<IntNode>(node)) {
        return ctx.create<IntType>();
    }
    if (isa<FloatNode>(node)) {
        return ctx.create<FloatType>();
    }
    if (isa<BreakNode>(node)||isa<ContinueNode>(node)|isa<EmptyNode>(node)) {
        return ctx.create<VoidType>();
    }

    if (auto binary = dyn_cast<BinaryNode>(node)) {
        auto lty = infer(binary->l);
        auto rty = infer(binary->r);
        if (binary -> kind == BinaryNode::And || binary -> kind == BinaryNode::Or) {
            if (isa<FloatType>(lty)) {
                auto zero = new FloatNode(0);
                zero -> type = ctx.create<FloatType>();
                auto ne = new BinaryNode(BinaryNode::Ne, binary->l, zero);
                ne -> type = ctx.create<IntType>();
                binary -> l = ne;
            }
            if (isa<FloatType>(rty)) {
                auto zero = new FloatNode(0);
                zero -> type = ctx.create<FloatType>();
                auto ne = new BinaryNode(BinaryNode::Ne, binary->r, zero);
                ne -> type = ctx.create<IntType>();
                binary -> r = ne;
            }
            return node -> type = ctx.create<IntType>();
        }

        // INT2FLOAT
        if (isa<FloatNode>(lty) && isa<IntType>(rty)) {
            binary -> r = new UnaryNode(UnaryNode::Int2Float, binary->r);
            rty = binary -> r -> type = ctx.create<FloatType>();
        }
        
        if (isa<IntType>(rty) && isa<FloatType>(lty)) {
            binary -> l = new UnaryNode(UnaryNode::Int2Float, binary->l);
            lty = binary -> l -> type = ctx.create<FloatType>();
        }

        std::set<decltype(BinaryNode::kind)> intops = {
            BinaryNode::And,BinaryNode::Or, BinaryNode::Eq,
            BinaryNode::Ne, BinaryNode::Le, BinaryNode::Lt,
        };

        if (isa<FloatType>(lty) && isa<FloatType>(rty) && !intops.count(binary->kind)) {
            return node -> type = ctx.create<FloatType>();
        }

        if (lty != rty) {
            std::cerr << "type mismatch" << std::endl;
            assert(false);
        }

        return node -> type = ctx.create<IntType>();
    }
    
    if (auto unary = dyn_cast<UnaryNode>(node)) {
        auto ty = infer(unary->node);
        assert(unary->kind != UnaryNode::Float2Int && unary->kind != UnaryNode::Int2Float);
        if (isa<FloatType>(ty) && unary->kind == UnaryNode::Minus) {
            return node -> type = ctx.create<FloatType>();
        }
        return node -> type = ctx.create<IntType>();
    }

    if (auto vardecl = dyn_cast<VarDeclNode>(node)) {
        assert(node->type);
        symbols[vardecl->name] = node->type;
        if (!vardecl->init) {
            return ctx.create<VoidType>();
        }
        if (vardecl->global && !vardecl->mut){
            vardecl->init->type = node->type;
        } else {
            auto ty = infer(vardecl->init);
            if (isa<IntType>(ty) && isa<FloatType>(vardecl->type)) {
                // Int2Float
                vardecl->init = new UnaryNode(UnaryNode::Int2Float, vardecl->init);
                vardecl->init->type = ctx.create<FloatType>();
                return ctx.create<VoidType>();
            }

            if (isa<FloatType>(ty) && isa<IntType>(vardecl->type)) {
                // Float2Int
                vardecl->init = new UnaryNode(UnaryNode::Float2Int, vardecl->init);
                vardecl->init->type = ctx.create<IntType>();
                return ctx.create<VoidType>();
            }

            if (ty != vardecl->type) {
                std::cerr << "bad assignment\n";
                assert(false);
            }
        }
        return ctx.create<VoidType>();
    }

    
        

    
    return nullptr;
    
}
