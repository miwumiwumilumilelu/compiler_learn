#ifndef ASTNODE_H
#define ASTNODE_H

#include <vector>
#include <string>
#include "Type.h"

namespace sys{

class ASTNode {
    const int id;
public:
    Type *type = nullptr;
    int getID() const { return id; }
    virtual ~ASTNode() {}
    ASTNode(int id): id(id) {}
};

template<class T,int NodeID>
class ASTNodeImpl : public ASTNode {
public:
  static bool classof(ASTNode *node) {
      return node->getID() == NodeID;
  }
    ASTNodeImpl(): ASTNode(NodeID) {}
};

class IntNode : public ASTNodeImpl<IntNode, __LINE__> {
public:
    int value;
    IntNode(int value): value(value) {}
};

class FloatNode : public ASTNodeImpl<FloatNode, __LINE__> {
public:
    float value;
    FloatNode(float value): value(value) {}
};

// a scoped block.
class BlockNode : public ASTNodeImpl<BlockNode, __LINE__> {
public:
    std::vector<ASTNode*> nodes;
    BlockNode(const decltype(nodes) &n): nodes(n) {}
    ~BlockNode() { for (auto node : nodes) delete node; }
};

// variable Dec.
class VarDeclNode : public ASTNodeImpl<VarDeclNode, __LINE__> {
public:
    std::string name;
    ASTNode *init; // can be nullptr.
    bool mut;
    bool global;

    VarDeclNode(const std::string &name, ASTNode *init, bool mut = true, bool global = false):
        name(name), init(init), mut(mut), global(global) {}
    ~VarDeclNode() { delete init; }
};

// variable Use.
class VarRefNode : public ASTNodeImpl<VarRefNode, __LINE__> {
public:
    std::string name;
    VarRefNode(const std::string &name): name(name) {}
};

// does not create a new scope,
// note that variables declared inside will still be in the outer scope.
class TransparentBlockNode : public ASTNodeImpl<TransparentBlockNode, __LINE__> {
public:
    std::vector<VarDeclNode*> nodes;
    TransparentBlockNode(const decltype(nodes) &n): nodes(n) {}
};

class BinaryNode : public ASTNodeImpl<BinaryNode, __LINE__> {
public:
  enum {
    Add, Sub, Mul, Div, Mod, And, Or,
    // >= and > Canonicalized.
    Eq, Ne, Le, Lt
  } kind;

  ASTNode *l, *r;

  BinaryNode(decltype(kind) k, ASTNode *l, ASTNode *r):
    kind(k), l(l), r(r) {}
  ~BinaryNode() { delete l; delete r; }
};

class UnaryNode : public ASTNodeImpl<UnaryNode, __LINE__> {
public:
  enum {
    Not, Minus, Float2Int, Int2Float
  } kind;
  ASTNode *node;
    UnaryNode(decltype(kind) k, ASTNode *node):
        kind(k), node(node) {}
    ~UnaryNode() { delete node; }
};

class FnDeclNode : public ASTNodeImpl<FnDeclNode, __LINE__> {
public:
    std::string name;
    std::vector<std::string> params;
    BlockNode *body;
    FnDeclNode(const std::string &name, const decltype(params) &params, BlockNode *body):
        name(name), params(params), body(body) {}
    ~FnDeclNode() { delete body; }
};

class ReturnNode : public ASTNodeImpl<ReturnNode, __LINE__> {
public:
    ASTNode *node; // can be nullptr.
    std::string func;
    ReturnNode(const std::string &func, ASTNode *node):
        func(func), node(node) {}
    ~ReturnNode() { delete node; }
};

class IfNode : public ASTNodeImpl<IfNode, __LINE__> {
public:
      ASTNode *cond, *ifso, *ifnot;

  IfNode(ASTNode *cond, ASTNode *ifso, ASTNode *ifnot):
    cond(cond), ifso(ifso), ifnot(ifnot) {}
  ~IfNode() { delete cond; delete ifso; delete ifnot; }
};

class WhileNode : public ASTNodeImpl<WhileNode, __LINE__> {
public:
    ASTNode *cond, *body;
    WhileNode(ASTNode *cond, ASTNode *body):
        cond(cond), body(body) {}
    ~WhileNode() { delete cond; delete body; }
};

class ForNode : public ASTNodeImpl<ForNode, __LINE__> {
public:
    ASTNode *init, *cond, *incr, *body;
    ForNode(ASTNode *init, ASTNode *cond, ASTNode *incr, ASTNode *body):
        init(init), cond(cond), incr(incr), body(body) {}
    ~ForNode() { delete init; delete cond; delete incr; delete body; }
};

class AssignNode : public ASTNodeImpl<AssignNode, __LINE__> {
public:
    ASTNode *l, *r;
    AssignNode(ASTNode *l, ASTNode *r):
        l(l), r(r) {}
    ~AssignNode() { delete l; delete r; }
};

class ConstArrayNode : public ASTNodeImpl<ConstArrayNode, __LINE__> {
public:
    union{
        int *vi;
        float *vf;
    };
    bool isFloat;
    ConstArrayNode(int *vi): vi(vi), isFloat(false) {}
    ConstArrayNode(float *vf): vf(vf), isFloat(true) {}
};

class LocalArrayNode : public ASTNodeImpl<LocalArrayNode, __LINE__> {
public:
    ASTNode **elements;
    LocalArrayNode(ASTNode **elements): elements(elements) {}
    ~LocalArrayNode() {
        // assume elements is null-terminated
        for (int i = 0; elements[i] != nullptr; i++) {
            delete elements[i];
        }
        delete[] elements;
    }
};

class ArrayAccessNode : public ASTNodeImpl<ArrayAccessNode, __LINE__> {
public:
    std::string array;
    std::vector<ASTNode*> indices;
    Type *arrayType = nullptr; // Filled during semantic analysis.
    ArrayAccessNode(const std::string &array, const std::vector<ASTNode*> &indices):
        array(array), indices(indices) {}
    ~ArrayAccessNode() {
        for (auto idx : indices) delete idx;
    }
};

class ArrayAssignNode : public ASTNodeImpl<ArrayAssignNode, __LINE__> {
public:
    std::string array;
    std::vector<ASTNode*> indices;
    ASTNode *value;
    Type *arrayType = nullptr; // Filled during semantic analysis.
    ArrayAssignNode(const std::string &array, const decltype(indices) &indices, ASTNode *value):
        array(array), indices(indices), value(value) {}
    ~ArrayAssignNode() {
        for (auto idx : indices) delete idx;
        delete value;
    }
};

class CallNode : public ASTNodeImpl<CallNode, __LINE__> {
public:
    std::string callee;
    std::vector<ASTNode*> args;
    CallNode(const std::string &callee, const decltype(args) &args):
        callee(callee), args(args) {}
    ~CallNode() { for (auto arg : args) delete arg; }
};

class BreakNode : public ASTNodeImpl<BreakNode, __LINE__> {};
class ContinueNode : public ASTNodeImpl<ContinueNode, __LINE__> {};
class EmptyNode : public ASTNodeImpl<EmptyNode, __LINE__> {};

}
#endif // ASTNODE_H