# 2025.12.9 后端codegen——IR Builder

参考LLVM 的 IRBuilder设计，但采用重复的8个模板函数重载来简化完美转发，并仅设计最小可用的单一Builder

实现从 AST 到 IR 转换。通过codegen链接前端，包括实现了 `Builder` 类用于管理指令插入，以及 `CodeGen` 类。并特别完成了二元表达式（`BinaryNode`），特别是实现了其中的and/or的短路求值转换逻辑



## **CodeGen.h:**

```c++
class Builder {
    BasicBlock *bb;
    BasicBlock::iterator at;
    bool init = false;
public:
  // Guards insertion point.
    struct Guard {
        Builder &builder;
        BasicBlock *bb;
        BasicBlock::iterator at;
    public:
        Guard(Builder &builder): builder(builder), bb(builder.bb), at(builder.at) {}
        ~Guard() { builder.bb = bb; builder.at = at; }
    };
```

通过基本块作为中间载体`bb->insert(at, op);`来插入指令，通过appendRegion和appendBlock来进行新区域、新基本块构建等

此处成员变量`init`来确定构建器是否定位完成

`Guard`类作为作用域守卫，用来保存和恢复Builder

```c++
    // We use 8 overloads because literal { ... } can't be deduced.
    template<class T>
    T *create() {
        assert(init);
        auto op = new T();
        bb->insert(at, op);
        return op;
    }

    template<class T>
    T *create(const std::vector<Value> &v) {
        assert(init);
        auto op = new T(v);
        bb->insert(at, op);
        return op;
    }

    template<class T>
    T *create(const std::vector<Attr*> &v) {
        assert(init);
        auto op = new T(v);
        bb->insert(at, op);
        return op;
    }

    template<class T>
    T *create(Value::Type resultTy) {
        assert(init);
        auto op = new T();
        bb->insert(at, op);
        return op;
    }

    template<class T>
    T *create(const std::vector<Value> &v, const std::vector<Attr*> &v2) {
        assert(init);
        auto op = new T(v, v2);
        bb->insert(at, op);
        return op;
    }

    template<class T>
    T *create(Value::Type resultTy, const std::vector<Attr*> &v) {
        assert(init);
        auto op = new T();
        bb->insert(at, op);
        return op;
    }

    template<class T>
    T *create(Value::Type resultTy, const std::vector<Value> &v, const std::vector<Attr*> &a) {
        assert(init);
        auto op = new T();
        bb->insert(at, op);
        return op;
    }

    template<class T>
    T *create(Value::Type resultTy, const std::vector<Value> &v) {
        assert(init);
        auto op = new T(resultTy, v);
        bb->insert(at, op);
        return op;
    }
```

重载8次，对应不同的输入

create函数入参考虑完全来创建对应T类型指令，replace同理

```c++
    // This shallow-copies operands, but deep copies attributes.
    Op *copy(Op *op);
```

浅拷贝操作数、深拷贝属性

因为`Value`是引用语义，所以浅拷贝（引用同一个定义）；`Attribute`需要进行引用计数而不仅仅是语义，所以需要深拷贝并增加引用计数

```c++
class CodeGen {
    using SymbolTable = std::map<std::string, Value>;
  
    ModuleOp *module;
    Builder builder;
    SymbolTable symbols;
    SymbolTable globals;
    void emit(ASTNode *node);
    Value emitExpr(ASTNode *node);
    Value emitBinary(BinaryNode *node);
    Value emitUnary(UnaryNode *node);

    int getSize(Type *ty);
public:
    class SemanticScope {
        CodeGen &cg;
        SymbolTable symbols;
    public:
        SemanticScope(CodeGen &cg) : cg(cg), symbols(cg.symbols) {}
        ~SemanticScope() { cg.symbols = symbols; }
    };

    CodeGen(ASTNode *node);
    // forbid copy, because symbol table is not copied.
    CodeGen(const CodeGen &other) = delete;
  
    ModuleOp *getModule() { return module; }
};
```

IR生成类`CodeGen`首先需要能定位模块，使用符号表存作用域、全局域的Value，并进行作用域管理

采用递归下降进行解析处理：如`emitBinary` 内部会递归调用 `emitExpr` 来生成左操作数和右操作数的代码，然后再计算出一条插入指令

这里注意

```cpp
// forbid copy, because symbol table is not copied.
CodeGen(const CodeGen &other) = delete;
```

CodeGen类**禁止拷贝构造 `CodeGen` 对象**，防止符号表被多个对象共享修改，防止Builder指向同一个插入点，导致 IR 生成混乱，并且防止多个对象获取同一个模块（需要进行独占）



## **CodeGen.cpp:**

```c++
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
```

对应实现CodeGen.h中定义的6个用于构建的成员函数

```c++
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
```

拷贝实现，并插入已定位的at位置

```c++
CodeGen::CodeGen(ASTNode *node): module(new ModuleOp()) {
    module->createFirstBlock();
    builder.setToRegionStart(module->getRegion());
    emit(node);
}
```

构造函数，从Module开始，以AST根节点为node开始进行递归下降

```c++
int CodeGen::getSize(Type *ty) {
    assert(ty);
    if (isa<IntType>(ty) || isa<FloatType>(ty))
        return 4;
    if (auto arrTy = dyn_cast<ArrayType>(ty))
        return getSize(arrTy->base) * arrTy->getSize();

    return 8;
}
```

类型大小获取：Int/Float——>4	array——>base * ArrayType->getSize()

```c++
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
```

对于emitBinary二元运算节点类，只需要对And/Or进行特殊处理

即

```c++
        //   l && r
        // becomes
        //   if (l)
        //     %1 = not_zero r
        //     store %1, %alloca
        //   else
        //     store 0, %alloca
        //   load %alloca
```

和

```c++
        //   l || r
        // becomes
        //   if (l)
        //     store 1, %alloca
        //   else
        //     %1 = not_zero r
        //     store %1, %alloca
        //   load %alloca
```

实现同理

其余的如大小比较、四则运算则只需要关注Value本身的类型是否同为int（否则都转为float）