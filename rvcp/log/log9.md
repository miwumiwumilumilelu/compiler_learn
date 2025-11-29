# 2025.11.29 后端codegen——定义IR静态结构

效仿Mlir的IR设计与LLVM IR，我计划实现一个支持SSA形式的IR，因此从SSA性质与Use-Def链（自底向上）入手编写；并且要考虑到从CFG中读取信息，因此设计理念偏向于CFGIR，并结合了Mlir中的嵌套机制



## src/codegen/OpBase.h

```c++
class Op;
class BasicBlock;

class Value {
public:
    Op *defining;
    enum Type {
        unit, i32, i64, f32, i128, f128
    };
    Value(){}; // uninitialized, for std::map
    Value(Op *from);

    bool operator==(Value x) const { return defining == x.defining; }
    bool operator!=(Value x) const { return defining != x.defining; }
    bool operator<(Value x) const { return defining < x.defining; }
    bool operator>(Value x) const { return defining > x.defining; }
    bool operator<=(Value x) const { return defining <= x.defining; }
    bool operator>=(Value x) const { return defining >= x.defining; }
};
```

对于一个值遵循SSA仅一次定义的性质`Op defining`，因此构造函数需要传入定义它的指令

这里支持f128和i128类型，并给出指令Def处的比较操作符重载，后续会对std::map内的Value进行活跃变量分析并以此来区分不同Value（SSA性质）



```c++
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

/// CFG Mutations
    BasicBlock *appendBlock();
    void dump(std::ostream &os, int depth);

    BasicBlock *insert(BasicBlock* at);
    BasicBlock *insertAfter(BasicBlock* at);
    void remove(BasicBlock* at);

    void insert(iterator at, BasicBlock *bb);
    void insertAfter(iterator at, BasicBlock *bb);
    void remove(iterator at);

/// CFG Analyses
    void updatePreds();
    void updateDoms();
    void updateDomFront();
    void updatePDoms();
    void updateLiveness();
    std::pair<BasicBlock*, BasicBlock*> moveTo(BasicBlock *insertionPoint);

    void erase();
    Region(Op *parent): parent(parent) {}
};
```

重头戏来了，在LLVM中我们将模块划分为函数，将函数划分为基本块，而mlir大大简化了这一层级结构，引入了区域，使得 IR 变成了递归嵌套的结构（Op -> Region -> BasicBlock -> Op）

这样只需要考虑Op和Region机制即可，这里使用**双向链表存储属于该区域的所有基本块**`std::list<BasicBlock*>`，方便在任意位置插入、删除或移动基本块，对Entry Block 和 Exit Block 进行获取等基础操作

然后就是整个CFG修改的基础操作

最后也是最重要的是在对CFG进行分析时考虑到，一个基本块肯定需要能查前驱后继，计算支配树、逆支配树以及支配边界（用于诸多优化Pass和析构ø函数），计算LiveIn、LiveOut集合（活跃变量区间、寄存器分配等需要）

```cpp
void dump(std::ostream &os, int depth);
void showLiveIn();
```

给出两个调试函数用来打印IR结构（depth），和活跃变量信息



```c++
class SimplifyCFG;
class BasicBlock {
  std::list<Op*> ops;
  Region *parent;
  Region::iterator place;
  // Note these are dominatORs, which mean `this` is dominatED by the elements.
  std::set<BasicBlock*> doms;
  // Dominance frontiers. `this` dominatES all blocks which are preds of the elements.
  std::set<BasicBlock*> domFront;
  BasicBlock *idom = nullptr;
  // Similarly, post dominators.
  std::set<BasicBlock*> pdoms;
  // Immediate post dominator.
  BasicBlock *ipdom = nullptr;
  // Variable (results of the ops) alive at the beginning of this block.
  std::set<Op*> liveIn;
  // Variable (results of the ops) alive at the end of this block.
  std::set<Op*> liveOut;

  friend class Region;
  friend class Op;
```

Op与BasicBlock也是使用list存，方便查询、插入和删除操作等

这里设置友元，使得其可以访问BasicBlock类的private成员

```c++
public:
	...
  const auto &getDominanceFrontier() const { return domFront; }
  const auto &getPDoms() const { return pdoms; }
  const auto &getLiveIn() const { return liveIn; }
  const auto &getLiveOut() const { return liveOut; }

  std::vector<Op*> getPhis() const;
  
  BasicBlock *getIdom() const { return idom; }
  BasicBlock *getIPdom() const { return ipdom; }
  BasicBlock *nextBlock() const;

  bool dominatedBy(const BasicBlock *bb) const;
  bool dominates(const BasicBlock *bb) const { return bb->dominatedBy(this); }
  bool postDominatedBy(const BasicBlock *bb) const { return pdoms.count(const_cast<BasicBlock*>(bb)); }
  bool postDominates(const BasicBlock *bb) const { return bb->pdoms.count(const_cast<BasicBlock*>(bb)); }
	...
};
```

Op和BasicBlock的操作，类比于上一层级关系



```cpp
class Attr {
  int refcnt = 0;

  friend class Op;
  friend class Builder;
public:
  const int attrid;
  Attr(int id): attrid(id) {}
  
  virtual ~Attr() {}
  virtual std::string toString() = 0;
  virtual Attr *clone() = 0;
};
```

管理指令静态属性（如整数常量值、函数名等）



```c++
class Op {
protected:
  std::set<Op*> uses;
  std::vector<Value> operands;
  std::vector<Region*> regions;
  std::vector<Attr*> attrs;
  BasicBlock *parent;
  BasicBlock::iterator place;
  Value::Type resultTy;

  friend class Builder;
  friend class BasicBlock;

  std::string opname;
  // This is for ease of writing macro.
  void setName(std::string name);
  void removeOperandUse(Op *op);

  static std::vector<Op*> toDelete;
```

Use-Def链，其中Def仅为1，Use可以有多处

支持移除UseOp

```c++
public:
  const int opid;
	...//Op操作，在使用中再增量开发
  template<class T>
  bool has() {
    for (auto x : attrs)
      if (isa<T>(x))
        return true;
    return false;
  }

  template<class T>
  T *get() {
    for (auto x : attrs)
      if (isa<T>(x))
        return cast<T>(x);
    assert(false);
  }

  template<class T>
  T *find() {
    for (auto x : attrs)
      if (isa<T>(x))
        return cast<T>(x);
    return nullptr;
  }

  template<class T>
  void remove() {
    for (auto it = attrs.begin(); it != attrs.end(); it++)
      if (isa<T>(*it)) {
        if (!--(*it)->refcnt)
          delete *it;
        attrs.erase(it);
        return;
      }
  }

  template<class T, class... Args>
  void add(Args... args) {
    auto attr = new T(std::forward<Args>(args)...);
    attr->refcnt++;
    attrs.push_back(attr);
  }

  template<class T>
  std::vector<Op*> findAll() {
    std::vector<Op*> result;
    if (isa<T>(this))
      result.push_back(this);

    for (auto region : getRegions())
      for (auto bb : region->getBlocks())
        for (auto x : bb->getOps()) {
          auto v = x->findAll<T>();
          std::copy(v.begin(), v.end(), std::back_inserter(result));
        }

    return result;
  }

  template<class T>
  T *getParentOp() {
    auto parent = getParentOp();
    while (!isa<T>(parent))
      parent = parent->getParentOp();

    return cast<T>(parent);
  }
};
```

设置OpID，需要为指令进行编号，这很重要，各个Pass中会用到，代码实质上就是字符串（如在公共代码提取Pass中用于后缀树找最大公共子串来减少跳转指令从而尽量避免流水线中断），若将每一个字符记录就会很麻烦且浪费内存，所以使用编号来简化标识

定义模板函数用于检查或获取依附于当前指令上的静态属性，属性存储在 `std::vector<Attr*> attrs` 中

* 对于确定属性的Op获取其T属性指针时，如果get不到，就直接assert，而不会return false

定义模板函数用于IR 结构遍历，进行向上和向下的类型化搜索

* 这里支持了指定类型的当前Op的父亲Op查找，指定类型的当前Op嵌套下的所有Op查找（使用递归遍历Op->region->bb->op）



```c++
inline std::ostream &operator<<(std::ostream &os, Op *op) {
  op->dump(os);
  return os;
}
```

重载了 cpp 标准输出流的 `<<` 运算符，调用dump()

```c++
template<class T, int OpID>
class OpImpl : public Op {
public:
  constexpr static int id = OpID;
  
  static bool classof(Op *op) {
    return op->opid == OpID;
  }

  OpImpl(Value::Type resultTy, const std::vector<Value> &values): Op(OpID, resultTy, values) {}
  OpImpl(Value::Type resultTy, const std::vector<Value> &values, const std::vector<Attr*> &attrs):
    Op(OpID, resultTy, values, attrs) {}
};
```

定义OpImpl实现类模板函数

支持验证ID——classof，检查传入的 `op` 的 `opid` 成员是否等于模板参数 `OpID`

```c++
template<class T, int AttrID>
class AttrImpl : public Attr {
public:
  static bool classof(Attr *attr) {
    return attr->attrid == AttrID;
  }

  AttrImpl(): Attr(AttrID) {}
};


}; //namespace
#endif
```

定义AttrImpl实现类模板函数



这套 IR 设计构建了一个SSA形式IR但区别于LLVM的不同层次架构的清亮编译器后端基础设施，方便后续编写复杂的优化Pass，此外对于特殊Pass需要返回来进行增量修改
