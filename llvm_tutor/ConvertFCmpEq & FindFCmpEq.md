# ConvertFCmpEq & FindFCmpEq

## ConvertFCmpEq & FindFCmpEq 使用 —— 检测危险浮点相等比较指令

### FindFCmpEq——分析Pass

**分析 Pass**，在代码中找出所有直接进行相等比较的浮点数运算

由于计算机浮点数表示法的 inherent 精度问题（舍入误差），直接使用 `==` 来判断两个浮点数是否相等是一种不健壮的编程实践，常常会导致意想不到的逻辑错误

而这个Pass遍历整个程序，找出所有这些“危险”的比较操作



这个 Pass 是后续 `ConvertFCmpEq`**转换 Pass** 的基础



```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -emit-llvm -S -Xclang -disable-O0-optnone -c ../inputs/input_for_fcmp_eq.c -o input_for_fcmp_eq.ll
clang: warning: argument unused during compilation: '-c' [-Wunused-command-line-argument]
```

这里有个善意的提醒：`clang: warning: argument unused during compilation: '-c' [-Wunused-command-line-argument]`

其并不影响文件.ll生成，只是警告

原因如下：

**`-S`**: 告诉 `clang`：“请编译代码，并生成**人类可读的汇编/IR 代码**（文本文件）”。当想生成 `.s`（汇编）或 `.ll`（LLVM IR）文件时，就会用这个标志

**`-c`**: 告诉 `clang`：“请只进行**编译和汇编**，不要进行链接”。这个标志通常用来生成**二进制的目标文件**（ `.o` 文件，在 `llvm-tutor` 的例子中是 `.bc` 文件）

**冲突点在于**：`-S` 标志已经隐含了“不要进行链接”的意思，因为它只要求生成文本格式的中间代码，这个过程本身就不涉及链接



```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt --load-pass-plugin ./lib/libFindFCmpEq.dylib --passes="print<find-fcmp-eq>" -disable-output input_for_fcmp_eq.ll
Floating-point equality comparisons in "sqrt_impl":
  %11 = fcmp oeq double %9, %10
Floating-point equality comparisons in "main":
  %9 = fcmp oeq double %8, 1.000000e+00
  %13 = fcmp oeq double %11, %12
  %19 = fcmp oeq double %17, %18
```



### ConvertFCmpEq——转换Pass

**ConvertFCmpEq** 过程是一种转换，它使用 FindFCmpEq 的分析结果，将直接浮点相等性比较指令转换为使用预先计算的舍入阈值的逻辑等效指令

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -emit-llvm -S -Xclang -disable-O0-optnone -c ../inputs/input_for_fcmp_eq.c -o input_for_fcmp_eq.ll
clang: warning: argument unused during compilation: '-c' [-Wunused-command-line-argument]

llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt --load-pass-plugin ./lib/libFindFCmpEq.dylib \
                             --load-pass-plugin ./lib/libConvertFCmpEq.dylib \
                             --passes=convert-fcmp-eq -S input_for_fcmp_eq.ll -o fcmp_eq_after_conversion.ll 

llvm-tutor/build on  main [?] via 🅒 base 
➜ cat fcmp_eq_after_conversion.ll  
```

因为 `ConvertFCmpEq` 在内部会向 `AnalysisManager` 请求 `FindFCmpEq` 的结果，所以 `FindFCmpEq` 必须先被注册，`AnalysisManager` 才知道有这个分析服务存在

可以看到@main其中一个比较转换前：

```
%cmp = fcmp oeq double %a, %b
```

转换后：

```
; 在 label %14 中，对应 if (b == 1.0) { if (a == b) return 1; }
	%15 = load double, ptr %2, align 8
  %16 = load double, ptr %3, align 8
  %17 = fsub double %15, %16
  %18 = bitcast double %17 to i64
  %19 = and i64 %18, 9223372036854775807
  %20 = bitcast i64 %19 to double
  %21 = fcmp olt double %20, 0x3CB0000000000000 
```

9223372036854775807：**最高位是 `0`**，其余 **63 位全是 `1`** ，即0x7FFFFFFFFFFFFFFF

根据 IEEE 754 标准，一个浮点数的**最高位是符号位**：`0` 代表正数，`1` 代表负数。

通过将一个数与 `0x7FFFFFFFFFFFFFFF` 进行按位与，我们实际上是在说：“**保持所有位不变，但强行将最高位（符号位）设置为 0**”

**前提是先bitcast进行位转换为整数，结合这个位与操作，这样就高效地实现了取绝对值**

得到abs(a - b)

最后进行机器最小精度比较：

`0x3CB0000000000000`：这是机器 epsilon的十六进制浮点数表示。它是一个非常小的正数，代表了计算机能区分的最小精度

`abs(a - b) < epsilon`



## ConvertFCmpEq & FindFCmpEq源码

### FindFCmpEq.h

`using Result = std::vector<llvm::FCmpInst *>;`返回`FCmpInst`指针

```c++
// Forward declarations
namespace llvm {

class FCmpInst;
class Function;
class Module;
class raw_ostream;

} // namespace llvm

```

获取 `FindFCmpEq` 的分析结果并将其打印出来

```c++
class FindFCmpEqPrinter : public llvm::PassInfoMixin<FindFCmpEqPrinter> {
public:
  explicit FindFCmpEqPrinter(llvm::raw_ostream &OutStream) : OS(OutStream){};

  llvm::PreservedAnalyses run(llvm::Function &Func,
                              llvm::FunctionAnalysisManager &FAM);

private:
  llvm::raw_ostream &OS;
};
```



### FindFCmpEq.cpp

```c++
FindFCmpEq::Result FindFCmpEq::run(Function &Func) {
  Result Comparisons;

  // 1. 遍历函数中的每一条指令
  for (Instruction &Inst : instructions(Func)) {
    // 2. 过滤：是不是浮点数比较指令？
    if (auto *FCmp = dyn_cast<FCmpInst>(&Inst)) {
      // 3. 过滤：是不是“相等”比较？
      if (FCmp->isEquality()) {
        // 4. 如果是，就把它加入到结果列表中
        Comparisons.push_back(FCmp);
      }
    }
  }

  return Comparisons; // 5. 返回找到的所有指令
}
```

打印Pass的协作：

```c++
PreservedAnalyses FindFCmpEqPrinter::run(Function &Func,
                                         FunctionAnalysisManager &FAM) {
  // 关键：向分析管理器请求 FindFCmpEq 的结果
  auto &Comparisons = FAM.getResult<FindFCmpEq>(Func);

  // 调用辅助函数打印结果
  printFCmpEqInstructions(OS, Func, Comparisons);
  return PreservedAnalyses::all();
}
```



### ConvertFCmpEq.h

```c++
// 关键：包含了 FindFCmpEq.h，表明了依赖关系
#include "FindFCmpEq.h" 

struct ConvertFCmpEq : llvm::PassInfoMixin<ConvertFCmpEq> {
  // Pass 主入口，与 Pass 管理器交互
  llvm::PreservedAnalyses run(llvm::Function &Func,
                              llvm::FunctionAnalysisManager &FAM);
  // 核心逻辑的辅助函数，直接接收分析结果
  bool run(llvm::Function &Func, const FindFCmpEq::Result &Comparisons);

  static bool isRequired() { return true; }
};
```



### ConvertFCmpEq.cpp

```c++
// 这是 Pass 的主入口
PreservedAnalyses ConvertFCmpEq::run(Function &Func,
                                     FunctionAnalysisManager &FAM) {
  // 1. 请求依赖：向分析管理器请求 FindFCmpEq 的分析结果
  auto &Comparisons = FAM.getResult<FindFCmpEq>(Func);
  // 2. 调用核心逻辑函数，并传入分析结果
  bool Modified = run(Func, Comparisons);
  // 3. 根据是否修改了代码，返回正确的 PreservedAnalyses
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// 这是核心逻辑函数convertFCmpEqInstruction的上层管理函数
bool ConvertFCmpEq::run(Function &Func,
                        const FindFCmpEq::Result &Comparisons) {
  bool Modified = false;
  // ... (跳过 optnone 函数) ...
  
  // 遍历 FindFCmpEq 找到的所有目标指令
  for (FCmpInst *FCmp : Comparisons) {
    // 对每一条指令执行转换
    if (convertFCmpEqInstruction(FCmp)) {
      ++FCmpEqConversionCount; // 更新统计
      Modified = true;
    }
  }
  return Modified;
}
```

此处跳过optnone详解：

```cpp
if (Func.hasFnAttribute(Attribute::OptimizeNone)) {
    LLVM_DEBUG(dbgs() << "Ignoring optnone-marked function \"" << Func.getName()
                      << "\"\n");
    Modified = false;
    }
```

**`hasFnAttribute(...)`**: 这是 `Function` 类的一个成员函数，它的作用是检查这个函数是否带有一个特定的**属性 (Attribute)**

**`Attribute::OptimizeNone`**: 这是一个枚举值，它代表的就是 LLVM IR 中的 `optnone` 属性。这个属性通常由 `clang` 在 `-O0`（无优化）编译级别下自动添加，它的含义是“请不要对这个函数进行任何优化”，即不进行else块内容，不进行IR修改







**核心转换逻辑——`convertFCmpEqInstruction` 函数：**



**再次检查入参是否为空，且是否是等于比较**

```c++
static FCmpInst *convertFCmpEqInstruction(FCmpInst *FCmp) noexcept {
  assert(FCmp && "The given fcmp instruction is null");

  if (!FCmp->isEquality()) {
    return nullptr;
  }
```

`noexcept` 是 C++11 引入的一个关键字，用于函数的声明中，用来指明这个函数**是否可能抛出异常**

分析了函数体并确定，一个 C++ 异常是不可能从这里抛出的，因此标记noexcept



**将原始的比较操作（`==` 或 `!=`）映射到新的比较操作上，对指令进行谓词更换**

```c++
	Value *LHS = FCmp->getOperand(0); // a
  Value *RHS = FCmp->getOperand(1); // b

  CmpInst::Predicate CmpPred = [FCmp] {
    switch (FCmp->getPredicate()) {
    case CmpInst::Predicate::FCMP_OEQ: // a == b
      return CmpInst::Predicate::FCMP_OLT; // -> abs(a-b) < epsilon
    case CmpInst::Predicate::FCMP_UEQ: // a != b
      return CmpInst::Predicate::FCMP_ULT; // -> abs(a-b) >= epsilon
    // ... (处理其他情况) ...
    }
  }();
```

`CmpInst::Predicate CmpPred = [FCmp] {...}();`

Lambda 的这种写法将“定义”和“调用”合并成了一步：

​	`[] { ... }` 这部分是 Lambda 表达式的主体。可以把它看作一个没有名字的、随用随创建的迷你函数

​	**`CmpInst::Predicate`**: 这是一个枚举类型，代表了 LLVM 中所有的比较谓词（如等于、大于、小于等）

​	在 Lambda 表达式 `{...}` 的末尾紧跟的一对圆括号 `()`，它的作用是**立即执行**我们刚刚定义的匿名函数

`NaN` 是一种特殊的浮点数值，用于表示无效的运算结果，比如 `0.0 / 0.0`。

- **无序 (Unordered)**: 如果比较的两个浮点数中，**至少有一个**是 `NaN`，那么这次比较就被认为是“无序”的
- **有序 (Ordered)**: 如果比较的两个浮点数**都不是** `NaN`，那么这次比较就是“有序”的

LLVM 的 `fcmp` 指令谓词通过第一个字母来区分这两种情况：

- **`o`**: 代表 **ordered**。只有在“有序”的情况下，才可能为 `true`
- **`u`**: 代表 **unordered**。只要是“无序”的，就**一定**为 `true`



**创建后续生成 IR 指令时需要用到的所有“原材料”**

```c++
	// 获取 LLVM 上下文和基本类型
  LLVMContext &Ctx = M->getContext();
  IntegerType *I64Ty = IntegerType::get(Ctx, 64);
  Type *DoubleTy = Type::getDoubleTy(Ctx);

  // 定义用于计算绝对值的“掩码”
  ConstantInt *SignMask = ConstantInt::get(I64Ty, ~(1L << 63));

  // 定义机器 Epsilon
  APInt EpsilonBits(64, 0x3CB0000000000000);
  Constant *EpsilonValue =
      ConstantFP::get(DoubleTy, EpsilonBits.bitsToDouble());
```

**`SignMask`**: 这是实现 `abs()` 的关键。`~(1L << 63)` 在 64 位系统上会生成一个**最高位是 `0`、其余 63 位全是 `1`** 的整数常量（即 `0x7FFFFFFFFFFFFFFF`）

**`EpsilonValue`**: `0x3CB0000000000000` 是双精度浮点数**机器 Epsilon** 的十六进制表示。代码通过 `APInt` (任意精度整数) 和 `ConstantFP` (浮点常量) 将这个十六进制值转换成一个 LLVM IR 中的浮点数常量



**生成abs(a-b)指令 & 进行fcmp指令修改**

```c++
// 将 IRBuilder 定位到旧的 fcmp 指令之前
  IRBuilder<> Builder(FCmp);

  // 生成计算 abs(a-b) 的指令序列
  auto *FSubInst = Builder.CreateFSub(LHS, RHS);           // %0 = fsub double %a, %b
  auto *CastToI64 = Builder.CreateBitCast(FSubInst, I64Ty); // %1 = bitcast double %0 to i64
  auto *AbsValue = Builder.CreateAnd(CastToI64, SignMask); // %2 = and i64 %1, 0x7f...
  auto *CastToDouble = Builder.CreateBitCast(AbsValue, DoubleTy); // %3 = bitcast i64 %2 to double

	// Rather than creating a new instruction, we'll just change the predicate and
  // operands of the existing fcmp instruction to match what we want.
  FCmp->setPredicate(CmpPred);
  FCmp->setOperand(0, CastToDouble);
  FCmp->setOperand(1, EpsilonValue);
  return FCmp;
```

**`FCmp->setPredicate(CmpPred)`**: 将旧指令的比较谓词（如 `oeq`）**修改**为我们在之前计算出的新谓词 `CmpPred`

**`FCmp->setOperand(...)`**: 将旧指令的操作数**替换**掉

- 原来的操作数是 `a` 和 `b`。
- 现在，第一个操作数被换成了我们刚刚计算出的 `abs(a-b)` (`CastToDouble`)。
- 第二个操作数被换成了机器 Epsilon (`EpsilonValue`)。
