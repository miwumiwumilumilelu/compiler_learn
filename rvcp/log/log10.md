# 2025.12.2 后端codegen——定义IR静态结构

上次已经编写了op和attr基类，现在进行具体实现

## **Attrs.h & Attrs.cpp** 

**字面量与标识符**

按照基类进行构造即可

- `NameAttr`: 存储函数名、全局变量名等字符串标识符

  ```c++
  class NameAttr : public AttrImpl<NameAttr, __LINE__> {
  public:
      std::string name;
      NameAttr(std::string name): name(name) {}
  
      std::string toString() override { return "<name = " + name + ">"; }
      NameAttr *clone() override { return new NameAttr(name); }
  };
  ```

- `IntAttr` / `FloatAttr`: 存储源代码中的整数和浮点数常量值

- `SizeAttr`: 存储内存操作的大小或对齐信息

**控制流拓扑 (CFG Structure):**

给出extern，暴露给.cpp，用来tostring不同的bbid，如bb0、bb1、bb2...

```c++
extern std::map<BasicBlock*, int> bbmap;
extern int bbid;
```

- `TargetAttr`: 标记无条件跳转或条件跳转（True 分支）的目标基本块
- `ElseAttr`: 标记条件跳转（False 分支）的目标基本块
- `FromAttr`: 专用于 Phi 节点，标记数据来源的前驱基本块

**数据段常量 (Data):**

```c++
class IntArrayAttr : public AttrImpl<IntArrayAttr, __LINE__> {
public:
    int *vi;
    // This is the number of elements in `vi`, rather than byte size,
    // For example, if vi = { 2, 3 }, then `size` is 2, rather than sizeof(int) * 2.
    int size;
    bool allZero;

    IntArrayAttr(int *vi, int size);

    std::string toString() override;
    IntArrayAttr *clone() override { return new IntArrayAttr(vi, size); }
};
```

构造函数实现在.cpp，判断构造的函数是否是allZero

```c++
// Attrs.cpp
IntArrayAttr::IntArrayAttr(int *vi, int size): vi(vi), size(size), allZero(true) {
  for (int i = 0; i < size; i++) {
    if (vi[i] != 0) {
      allZero = false;
      break;
    }
  }
}
```

- `IntArrayAttr` / `FloatArrayAttr`: 存储数组初始化数据（如 `int arr[] = {1, 2}`），支持全零优化标记。

**函数元数据 (Function Metadata):**

标记类可以不用构造函数

- `ImpureAttr`: 标记函数具有副作用（不可被 DCE 删除）

- `AtMostOnceAttr`: 标记仅执行一次的代码块

- `ArgCountAttr`：参数计数

- `CallerAttr`: 维护调用图信息，记录调用者列表，辅助内联优化

  ```c++
  class CallerAttr : public AttrImpl<CallerAttr, __LINE__> {
  public:
      // The functions in `callers` actually calls the function with this attribute.
      // For example,
      //    func <name = f> <caller = g, h>
      // means `f` is called by `g` and `h`.
      std::vector<std::string> callers;
      CallerAttr(const std::vector<std::string> &callers): callers(callers) {}
      CallerAttr() {} // empty callers
      std::string toString() override;
      CallerAttr *clone() override { return new CallerAttr(callers); }
  };
  ```

**高级优化分析 (Optimization Analysis):**

- **`AliasAttr` (关键)**: 存储别名分析结果。描述指针的基址（`AllocaOp` 或 `GlobalOp`）和偏移量，提供 `mustAlias`, `neverAlias` 等接口，是内存优化的基础。

  ```c++
  class AliasAttr : public AttrImpl<AliasAttr, __LINE__> {
  public:
    std::map<Op*, std::vector<int>> location;
    bool unknown;
  
    AliasAttr(): unknown(true) {}
    AliasAttr(Op *base, int offset): location({ { base, { offset } } }), unknown(false) {}
    AliasAttr(const decltype(location) &location): location(location), unknown(false) {}
  
    // Returns true if changed.
    bool add(Op *base, int offset);
    bool addAll(const AliasAttr *other);
    bool mustAlias(const AliasAttr *other) const;
    bool neverAlias(const AliasAttr *other) const;
    bool mayAlias(const AliasAttr *other) const;
    std::string toString() override;
    AliasAttr *clone() override { return unknown ? new AliasAttr() : new AliasAttr(location); }
  };
  ```

  成员函数详细实现在Attrs.cpp中

- `RangeAttr`: 存储整数值域 `[low, high]`，用于值域传播和分支消除

- `IncreaseAttr`: 存储归纳变量的演化规律（多项式系数），用于循环优化（强度削减）

- `VariantAttr`: 标记循环变式，辅助循环不变量外提

- `FPAttr`: 浮点栈分配标记

  ```c++
  // Marks whether an alloca is floating point.
  // This can't be deduced by return value because it's always i64.
  class FPAttr : public AttrImpl<FPAttr, __LINE__> {
  public:
      FPAttr() {}
  
      std::string toString() override { return "<fp>"; }
      FPAttr *clone() override { return new FPAttr; }
  };
  ```

  区分于`FloatArrayAttr`，标记属性，用于表示一个内存分配(alloca)是否是浮点类型，它不存储实际的浮点数值

- `DimensionAttr`: 数组维度信息



文件末尾定义一组宏（如 `V(op)`, `TARGET(op)`, `ALIAS(op)`），提供了简洁的语法来从指令中提取特定属性值，提升 Pass 开发的效率

```c++
#define V(op) (op)->get<IntAttr>()->value
#define F(op) (op)->get<FloatAttr>()->value
#define SIZE(op) (op)->get<SizeAttr>()->value
#define NAME(op) (op)->get<NameAttr>()->name
#define TARGET(op) (op)->get<TargetAttr>()->bb
#define ELSE(op) (op)->get<ElseAttr>()->bb
#define CALLER(op) (op)->get<CallerAttr>()->callers
#define ALIAS(op) (op)->get<AliasAttr>()
#define RANGE(op) (op)->get<RangeAttr>()->range
#define FROM(attr) cast<FromAttr>(attr)->bb
#define INCR(op) (op)->get<IncreaseAttr>()
#define DIM(op) (op)->get<DimensionAttr>()->dims
```

注意此处op作为整体考虑，带有括号，否则有顺序优先级问题（op实际是字符串，由多个字符组成）



## Ops.h

```c++
#define OPBASE(ValueTy, Ty) \
  class Ty : public OpImpl<Ty, __LINE__> { \
  public: \
    explicit Ty(const std::vector<Value> &values): OpImpl(ValueTy, values) { \
      setName(#Ty); \
    } \
    Ty(): OpImpl(ValueTy, {}) { \
      setName(#Ty); \
    } \
    explicit Ty(const std::vector<Attr*> &attrs): OpImpl(ValueTy, {}, attrs) { \
      setName(#Ty); \
    } \
    Ty(const std::vector<Value> &values, const std::vector<Attr*> &attrs): OpImpl(ValueTy, values, attrs) { \
      setName(#Ty); \
    } \
  }

// Ops that must be explicitly set a result type.
#define OPE(Ty) \
  class Ty : public OpImpl<Ty, __LINE__> { \
  public: \
    Ty(Value::Type resultTy, const std::vector<Value> &values): OpImpl(resultTy, values) { \
      setName(#Ty); \
    } \
    explicit Ty(Value::Type resultTy): OpImpl(resultTy, {}) { \
      setName(#Ty); \
    } \
    Ty(Value::Type resultTy, const std::vector<Attr*> &attrs): OpImpl(resultTy, {}, attrs) { \
      setName(#Ty); \
    } \
    Ty(Value::Type resultTy, const std::vector<Value> &values, const std::vector<Attr*> &attrs): OpImpl(resultTy, values, attrs) { \
      setName(#Ty); \
    } \
  }

#define OP(Ty) OPBASE(Value::i32, Ty)
#define OPF(Ty) OPBASE(Value::f32, Ty)
#define OPL(Ty) OPBASE(Value::i64, Ty)
#define OPV(Ty) OPBASE(Value::i128, Ty)
```

OPBASE——定义固定返回类型的指令(一般指令) ；

OPE——定义显式返回类型的指令（如LoadOp等不确定返回值的指令），创建它时必须显式告诉它返回什么类型;

`ValueTy`: 该指令固定的返回值类型;

`Ty`: 指令类的名字（例如 `AddIOp`）;

自动生成 4 个构造函数：覆盖了“仅操作数”、“无参数”、“仅属性”、“操作数+属性”这四种初始化情况

然后就是各种类型的Op进行列举

这里注意最后`#undef OP`，取消之前OP宏定义，避免命名冲突



**这里attribute的储存方式不是很好，还是拿个map<string, Attr*>好一些**

**然后phi的attribute设计也很坏，最好用一个attribute而不是一大堆**