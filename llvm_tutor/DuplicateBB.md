# DuplicateBB 

## DuplicateBB Pass使用 —— 拆分基本块为if-then-else结构

此过程将复制模块中的所有基本块，但不包括那些没有可到达整数值的基本块（通过 **RIV** 过程识别）。此类基本块的一个例子是函数中的入口块，该函数：

- 不接受任何参数
- 嵌入在未定义全局值的模块中

基本块的复制方式是，首先插入一个 `if-then-else` 结构，然后将原始基本块中的所有指令（ ø节点除外）克隆到两个新的基本块（原始基本块的克隆）中。`if-then-else` 构造被引入作为一种非平凡的机制，它决定了接下来走哪一个克隆的基本块以进行分支。此条件等同于：

```
if (var == 0)
	goto clone1
else 
	goto clone2
```

即

```c++
BEFORE:                     AFTER:
-------                     ------
                              [ if-then-else ]
             DuplicateBB           /  \
[ BB ]      ------------>   [clone 1] [clone 2]
                                   \  /
                                 [ tail ]

LEGEND:
-------
[BB]           - the original basic block
[if-then-else] - a new basic block that contains the if-then-else statement (inserted by DuplicateBB)
[clone 1|2]    - two new basic blocks that are clones of BB (inserted by DuplicateBB)
[tail]         - the new basic block that merges [clone 1] and [clone 2] (inserted by DuplicateBB)
```

DuplicateBB 用 4 个新的基本块取代BB，是LLVM中**`SplitBlockAndInsertIfThenElse`** 的精心包装

`llvm-project/llvm/include/llvm/Transforms/Utils/BasicBlockUtils.h`

```c++
/// SplitBlockAndInsertIfThenElse is similar to SplitBlockAndInsertIfThen,
/// but also creates the ElseBlock.
/// Before:
///   Head
///   SplitBefore
///   Tail
/// After:
///   Head
///   if (Cond)
///     ThenBlock
///   else
///     ElseBlock
///   SplitBefore
///   Tail
///
/// Updates DT if given.
LLVM_ABI void SplitBlockAndInsertIfThenElse(
    Value *Cond, BasicBlock::iterator SplitBefore, Instruction **ThenTerm,
    Instruction **ElseTerm, MDNode *BranchWeights = nullptr,
    DomTreeUpdater *DTU = nullptr, LoopInfo *LI = nullptr);
```



**Run the Pass**

```sh
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -emit-llvm -S -O1 ../inputs/input_for_duplicate_bb.c -o input_for_duplicate_bb.ll
```

此时

```c++
llvm-tutor/build on  main [?] via 🅒 base 
➜ cat ../inputs/input_for_duplicate_bb.c              
//=============================================================================
// FILE:
//      input_for_duplicate_bb.c
//
// DESCRIPTION:
//      Sample input file for the DuplicateBB pass.
//
// License: MIT
//=============================================================================
int foo(int arg_1) { return 1; }

///cat .ll文件：
define noundef i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  ret i32 1
}
```

这里只有一个基本块（入口块），并且 `foo` 接受一个参数（这意味着 **RIV** 的结果将是一个非空集合)

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libRIV.dylib -load-pass-plugin ./lib/libDuplicateBB.dylib -passes=duplicate-bb -S input_for_duplicate_bb.ll -o duplicate.ll  

llvm-tutor/build on  main [?] via 🅒 base 
➜ cat duplicate.ll
define noundef i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
lt-if-then-else-0:
  %1 = icmp eq i32 %0, 0
  br i1 %1, label %lt-clone-1-0, label %lt-clone-2-0

lt-clone-1-0:                                     ; preds = %lt-if-then-else-0
  br label %lt-tail-0

lt-clone-2-0:                                     ; preds = %lt-if-then-else-0
  br label %lt-tail-0

lt-tail-0:                                        ; preds = %lt-clone-2-0, %lt-clone-1-0
  ret i32 1
}
```

有四个基本块。所有新的基本块都以原始基本块的数字ID结尾（本例中为 `0` ）。`lt-if-then-else-0` 包含新的 `if-then-else` 条件语句。 `clone-1-0` 和 `clone-2-0` 是 `foo` 中原始基本块的克隆。 `lt-tail-0` 是合并 `clone-1-0` 和 `clone-2-0` 所需的额外基本块

**Pass处理前：**

```
[入口] --> [基本块: 仅包含 ret i32 1 指令] --> [函数返回]
```

**Pass处理后：**

```
[lt-if-then-else-0] 
     | 
     v 
  条件判断 
     | 
     | (icmp eq i32 %0, 0) 
     | 
     +------------------+
     | (条件为真)        | (条件为假)        
     v                  v                  
[lt-clone-1-0]      [lt-clone-2-0]    
     |                  |
     +--------+---------+
              |
              v
        [lt-tail-0: ret i32 1]
```



## DuplicateBB 源码

### .h

```c++
using BBToSingleRIVMap =
    std::vector<std::tuple<llvm::BasicBlock *, llvm::Value *>>;

using ValueToPhiMap = std::map<llvm::Value *, llvm::Value *>;
```

* `BBToSingleRIVMap`——将 BasicBlock BB 映射到一个整数值（该值定义在另一个 BasicBlock 中），且该值在 BB 中可达。当克隆 BB 时，BB 映射到的值用于 `if-then-else` 结构

* `ValueToPhiMap`——在复制前将一个值映射到一个Phi节点，该节点在复制/克隆后将相应的值合并

  当 `cloneBB` 函数复制一个基本块时，原基本块中的一条指令（比如 `%orig_val = add i32 %a, %b`）会被克隆成两条新指令（`%clone1_val` 和 `%clone2_val`）

  在合并块（`tail` 块）中，Pass 必须创建一个 `PHINode`（`%phi_val`）来合并这两个值

  这个 `ValueToPhiMap` (`ReMapper`) 就被用来存储这个**映射关系**：

  - **键**：`%orig_val`（指向原始指令的指针）
  - **值**：`%phi_val`（指向新创建的 `PHINode` 的指针）



` static bool isRequired() { return true; }`

如果 isRequired 返回 true，则对带有 optnone LLVM 属性修饰的函数将**跳过此步骤**。注意 clang -O0 会将所有函数修饰为 optnone。



```c++
namespace llvm {
class RandomNumberGenerator;
} // namespace llvm
```

在llvm空间声明一个类

`llvm::RandomNumberGenerator`

` std::unique_ptr<llvm::RandomNumberGenerator> pRNG;`

定义一个随机生成器



### .cpp

**run**

```c++
PreservedAnalyses DuplicateBB::run(llvm::Function &F,
                                   llvm::FunctionAnalysisManager &FAM) {
  // 初始化随机数生成器 (pRNG)，用于后续的随机选择
  if (!pRNG)
    pRNG = F.getParent()->createRNG("duplicate-bb");
  
  BBToSingleRIVMap Targets = findBBsToDuplicate(F, FAM.getResult<RIV>(F));

  // This map is used to keep track of the new bindings. Otherwise, the
  // information from RIV will become obsolete.
  ValueToPhiMap ReMapper;

  // Duplicate
  for (auto &BB_Ctx : Targets) {
    cloneBB(*std::get<0>(BB_Ctx), std::get<1>(BB_Ctx), ReMapper);
  }

  DuplicateBBCountStats = DuplicateBBCount;
  return (Targets.empty() ? llvm::PreservedAnalyses::all()
                          : llvm::PreservedAnalyses::none());
}
```

向FAM请求RIV Pass处理的结果，通过`findBBsToDuplicate`找出所有可以进行duplicate的基本块

`ReMapper` 会在多个 cloneBB 调用之间共享和传递

`std::get<0>` 获取元组的第一个元素 (BasicBlock*) 

`std::get<1>` 获取元组的第二个元素 (Value - 用作 if 条件的变量)



**findBBsToDuplicate**

```c++
DuplicateBB::BBToSingleRIVMap
DuplicateBB::findBBsToDuplicate(Function &F, const RIV::Result &RIVResult) {
  BBToSingleRIVMap BlocksToDuplicate;

  // 遍历函数中的所有基本块
  for (BasicBlock &BB : F) {
    // 过滤 1：跳过用于异常处理的 'landing pad' 块
    if (BB.isLandingPad())
      continue;

    // 过滤 2：获取 RIV 结果。如果 RIV 集合为空，则不能复制
    // (因为没有变量可用于创建 if 条件)
    auto const &ReachableValues = RIVResult.lookup(&BB);
    if (0 == ReachableValues.size()) {
      continue;
    }

    // 随机选择：从 RIV 集合中随机挑选一个变量
    auto Iter = ReachableValues.begin();
    std::uniform_int_distribution<> Dist(0, ReachableValues.size() - 1);
    std::advance(Iter, Dist(*pRNG)); // pRNG 在 run 函数中初始化

    // 过滤 3：如果随机选中的是全局变量，就跳过
    // (因为全局变量通常是常量，会导致 if(0==0) 这样的无效混淆)
    if (dyn_cast<GlobalValue>(*Iter)) {
      continue;
    }
    
    // 目标合格，加入“工作清单”
    BlocksToDuplicate.emplace_back(&BB, *Iter);
  }
  return BlocksToDuplicate;
}
```

`ReachableValues`是当前查找的BB对应的RIVResult，即可达的Value

随机选择思路：先取BB对应的所有可达Value的begin起始位置，然后在`[0, size-1]`这个区间对应容器中所有元素的有效索引，最后把它应用到 `pRNG` 这个“随机数引擎”上，产生随机值Dist，Iter = Iter+Dist 即可，此时**随机移动了Iter**

向下转型全局变量指针，检查能否成功

`emplace_back` —— 是 `std::vector` 的一个成员函数，意思是“在末尾（back）就地（emplace）构造一个新元素”



**cloneBB**

```c++
  // Don't duplicate Phi nodes - start right after them
  BasicBlock::iterator BBHead = BB.getFirstNonPHIIt();

  // Create the condition for 'if-then-else'
  IRBuilder<> Builder(&*BBHead);
  Value *Cond = Builder.CreateIsNull(
      ReMapper.count(ContextValue) ? ReMapper[ContextValue] : ContextValue);
```

&*BBHead——第一条非ø指令，此时BBHead前才可插入指令

BBHead 是迭代器，需要解引用来指向BasicBlock

检查 `ContextValue`是否在 `ReMapper`映射表中存在，即是不是上次基本块克隆后的产物

（`ReMapper` 扮演着一个“查找与替换”表的角色。它记录了所有在**上一次** `cloneBB` 调用中被 `PHINode` 替换掉的旧 `Value`）

- 如果存在：就必须用它对应的新 PHI 节点 ，返回映射后的值 `ReMapper[ContextValue]`
- 如果不存在：返回原始值 `ContextValue`

```cpp
///举例：
BB1:
  %v1 = add i32 1, 2  ; %v1 是一个 Value
  br label %BB2
  
BB2:
  ; 假设 RIV 分析确定 %v1 在 BB2 是可达的
  %v2 = add i32 %v1, 10
  ret i32 %v2
  
  
/*
第1步: cloneBB(BB1, %ctx1, ReMapper)
cloneBB 函数开始执行，ReMapper 此时是空的。
函数在 BB1 中克隆指令。当它遇到 Instr = %v1 = add i32 1, 2：
它创建了两个克隆体：%v1.clone1 和 %v1.clone2。
它创建了一个 PHINode 来合并它们：%phi1 = phi i32 [ %v1.clone1, ... ], [ %v1.clone2, ... ]
它用 %phi1 替换掉了 tail-1 块中的 %v1。
关键：它更新了 ReMapper！ ReMapper[&Instr] = Phi;
ReMapper 现在包含：{ %v1 -> %phi1 }
cloneBB for BB1 结束。ReMapper 被保留，并传递给下一次调用

第2步: cloneBB(BB2, %v1, ReMapper)
cloneBB 函数再次启动。
BB 是 BB2。
ContextValue 是 %v1。
ReMapper 不是空的，它包含 { %v1 -> %phi1 }

第3步:
通过
Value *Cond = Builder.CreateIsNull(
      ReMapper.count(ContextValue) ? ReMapper[ContextValue] : ContextValue);
替换变量：
ContextValue 是 %v1
执行三元运算符：
ReMapper.count(%v1)：
“ReMapper 中有 %v1 这个键吗？”
“有，它在第1步被放进去了！”
count 返回 1，布尔值为 true。
因为结果是 true，所以我们执行 ReMapper[ContextValue]：
ReMapper[%v1] 返回它对应的值，即 %phi1
Builder.CreateIsNull(%phi1)：
创建的指令是 icmp eq i32 %phi1, 0
Cond 被设置为这个新指令的结果。
*/
```

```c++
  Instruction *ThenTerm = nullptr;
  Instruction *ElseTerm = nullptr;
  SplitBlockAndInsertIfThenElse(Cond, &*BBHead, &ThenTerm, &ElseTerm);
```

**`SplitBlockAndInsertIfThenElse` **

- 它在 `BBHead` 处将 `BB` **一分为二**
- **`BB` 的前半部分**（包含 PHI）现在成了 `if-then-else` 块。`Builder` 插入的 `if (Cond)` 成为了这个块新的终结符
- **`BB` 的后半部分**（从 `BBHead` 开始的所有指令）被移动到一个**新的 `tail` 块**中
- 它创建了两个**空**的块（`clone-1` 和 `clone-2`），并让它们都跳转到 `tail` 块
- `ThenTerm` 和 `ElseTerm` 现在分别指向 `clone-1` 和 `clone-2` 块中的 `br` (跳转) 指令

```c++
	// 5. 获取 tail 块的指针
  BasicBlock *Tail = ThenTerm->getSuccessor(0);

  // 6. (可选) 给新块命名，方便调试
  std::string DuplicatedBBId = std::to_string(DuplicateBBCount);
  ThenTerm->getParent()->setName("lt-clone-1-" + DuplicatedBBId); // then 块
  ElseTerm->getParent()->setName("lt-clone-2-" + DuplicatedBBId); // else 块
  Tail->setName("lt-tail-" + DuplicatedBBId);                       // tail 块
  // if 块 (即 BB 的前半部分)
  ThenTerm->getParent()->getSinglePredecessor()->setName("lt-if-then-else-" +
                                                         DuplicatedBBId);
```

**接下来填充 `clone` 块**

```c++
	// 7. 准备三个局部的 "查找替换" 映射
  ValueToValueMapTy TailVMap, ThenVMap, ElseVMap;
  // 准备一个列表，存放 tail 中不再需要的指令
  SmallVector<Instruction *, 8> ToRemove;

  // 8. 遍历 tail 块中的每一条原始指令
  for (auto IIT = Tail->begin(), IE = Tail->end(); IIT != IE; ++IIT) {
    Instruction &Instr = *IIT;

    // 9. 跳过终结符 (我们只克隆“计算”指令)
    if (Instr.isTerminator()) {
      // 但我们仍需更新终结符的操作数，以防它使用了 tail 块中某个即将被 PHI 替换的指令
      RemapInstruction(&Instr, TailVMap, RF_IgnoreMissingLocals);
      continue;
    }

    
    
    // 10. 克隆指令，一式两份
    Instruction *ThenClone = Instr.clone(), *ElseClone = Instr.clone();

    // 11. 处理 ThenClone (克隆体1)
    // 更新 ThenClone 的操作数。
    // 比如 Instr 是 "add %a, %b"，而 %a, %b 也是 tail 里的指令。
    // RemapInstruction 会在 ThenVMap 里查找 %a 和 %b 对应的克隆体
    // "add %a.clone, %b.clone"
    RemapInstruction(ThenClone, ThenVMap, RF_IgnoreMissingLocals);
    // 将克隆体插入到 clone-1 块的末尾 (终结符之前)
    ThenClone->insertBefore(ThenTerm->getIterator());
    // 记录映射：原始 Instr -> ThenClone
    ThenVMap[&Instr] = ThenClone;

    // 12. 处理 ElseClone (克隆体2)，同上
    RemapInstruction(ElseClone, ElseVMap, RF_IgnoreMissingLocals);
    ElseClone->insertBefore(ElseTerm->getIterator());
    ElseVMap[&Instr] = ElseClone;
```

此时，`clone-1` 和 `clone-2` 已经被填充了。但 `tail` 块里的原始指令 `Instr` 还在。我们如何处理它？

**最后用 PHI 替换 `tail`**

```c++
		// 13. 检查 Instr 是否产生值
    if (ThenClone->getType()->isVoidTy()) {
      // 情况 A: Instr 不产生值 (如 store, void call)
      // 既然 clone-1 和 clone-2 都有了，tail 里的原始指令就没用了。
      ToRemove.push_back(&Instr);
      continue; // 继续下一条指令
    }

    // 情况 B: Instr 产生值 (如 add, load)
    // 我们必须在 tail 块中合并 ThenClone 和 ElseClone 的结果。

    // 14. 创建一个新的 PHI 节点
    PHINode *Phi = PHINode::Create(ThenClone->getType(), 2);
    
    // 15. 告诉 PHI 节点：
    // 如果从 clone-1 块来，值是 ThenClone
    Phi->addIncoming(ThenClone, ThenTerm->getParent());
    // 如果从 clone-2 块来，值是 ElseClone
    Phi->addIncoming(ElseClone, ElseTerm->getParent());

    // 16. 记录映射 (局部 + 全局)
    // TailVMap: 用于阶段2的第9步，修复 tail 的终结符
    TailVMap[&Instr] = Phi; 
    // ReMapper: 全局映射，告诉函数中任何其他块：
    // "Instr 的结果现在由 Phi 代表"
    ReMapper[&Instr] = Phi; 

    // 17. 【关键】在 tail 块中，用 PHI 节点替换掉原始指令 Instr
    ReplaceInstWithInst(Tail, IIT, Phi);
  } // for 循环结束
```

`for` 循环跑完后，`tail` 块已经被“掏空”了。它所有的“计算”指令，要么被 `PHINode` 替换了（如果产生值），要么被标记为 `ToRemove`（如果不产生值）

**删除 `tail` 块中那些被标记为 `ToRemove` 的无用指令:**

```c++
	// 18. 遍历 ToRemove 列表
  for (auto *I : ToRemove)
    I->eraseFromParent(); // 从 tail 块中彻底删除

  // 19. 更新统计
  ++DuplicateBBCount;
}
```



### 举例

假设我们有以下 LLVM IR 函数。`entry` 块定义了一个值 `%val`，`work` 块使用了这个值

```
define i32 @foo(i32 %arg) {
entry:
  ; %val 将是 'work' 块的 RIV (可达整数值)
  %val = add i32 %arg, 1
  br label %work

work:
  ; BBHead 将指向这里
  %mul = mul i32 %val, 2     ; 指令1 (产生值)
  call void @print(i32 %mul) ; 指令2 (void, 不产生值)
  %add = add i32 %mul, 10    ; 指令3 (产生值)
  ret i32 %add               ; 指令4 (终结符)
}
```



 **`DuplicateBB::run` (Pass 入口)**

`run` 函数开始执行

```c++
// 1. 初始化随机数生成器
if (!pRNG)
  pRNG = F.getParent()->createRNG("duplicate-bb");

// 2. 获取 RIV 分析结果 (我们假设 RIV Pass 已经运行)
// FAM.getResult<RIV>(F)

// 3. 查找要克隆的目标
BBToSingleRIVMap Targets = findBBsToDuplicate(F, FAM.getResult<RIV>(F));
```



**`findBBsToDuplicate` (查找目标)**

`run` 函数调用 `findBBsToDuplicate`，传入 `@foo` 函数

```cpp
DuplicateBB::BBToSingleRIVMap
DuplicateBB::findBBsToDuplicate(Function &F, const RIV::Result &RIVResult) {
  BBToSingleRIVMap BlocksToDuplicate; // 1. 创建一个空的工作列表

  for (BasicBlock &BB : F) { // 2. 遍历 @foo 的所有基本块
    // 循环 1: BB = "entry"
    // ... 假设 "entry" 没有 RIV， 'continue' ...

    // 循环 2: BB = "work"
    // ... 'isLandingPad()' -> false ...

    // 3. 查找 "work" 块的 RIV
    auto const &ReachableValues = RIVResult.lookup(&BB);
    // 假设 RIVResult 返回 { %val }
    // ReachableValuesCount = 1

    // 4. RIV 集合不为空 (1 > 0)
    if (0 == ReachableValuesCount) { /* ... */ }

    // 5. 随机选择一个 RIV
    auto Iter = ReachableValues.begin(); // Iter 指向 %val
    std::uniform_int_distribution<> Dist(0, 0); // (0, size-1)
    std::advance(Iter, Dist(*pRNG)); // 随机数必须是0, Iter 仍然指向 %val

    // 6. 检查是否为 GlobalValue
    if (dyn_cast<GlobalValue>(*Iter)) {
      // dyn_cast<GlobalValue>(%val) -> false, 因为 %val 是 'entry' 块的指令
      // ... 'continue' 被跳过 ...
    }

    // 7. 将任务添加到工作列表
    BlocksToDuplicate.emplace_back(&BB, *Iter);
    // BlocksToDuplicate 现在是: { <&BB "work">, <Value* %val> }
  }

  return BlocksToDuplicate; // 8. 返回工作列表
}
```



 **`DuplicateBB::run` (继续执行)**

`run` 函数拿到了 `Targets` 列表

```c++
// 4. `Targets` = { <&BB "work">, <Value* %val> }

// 5. 创建一个空的全局值替换表
ValueToPhiMap ReMapper; // ReMapper = {}

// 6. 遍历工作列表
for (auto &BB_Ctx : Targets) {
  // 循环 1:
  //   BB_Ctx.get<0>() 是 "work" 块的指针
  //   BB_Ctx.get<1>() 是 %val
  // 7. 调用核心克隆函数
  cloneBB(*std::get<0>(BB_Ctx), std::get<1>(BB_Ctx), ReMapper);
}

// 8. 更新统计数据
DuplicateBBCountStats = DuplicateBBCount; // 假设 cloneBB 将其设为 1
return (Targets.empty() ? ... : PreservedAnalyses::none()); // 返回 none()
```



**`DuplicateBB::cloneBB` (核心手术)**

这是最关键的部分。`cloneBB` 被调用，参数为：

- `BB` = "work"
- `ContextValue` = `%val`
- `ReMapper` = `{}` (空 map)

```c++
void DuplicateBB::cloneBB(BasicBlock &BB, Value *ContextValue,
                          ValueToPhiMap &ReMapper) {
  // 1. 找到插入点 (第一条非 PHI 指令)
  // "work" 块没有 PHI, BBHead 指向 %mul = mul i32 %val, 2
  BasicBlock::iterator BBHead = BB.getFirstNonPHIIt();

  // 2. 创建 IR 构造器，设置插入点为 BBHead 之前
  IRBuilder<> Builder(&*BBHead);

  // 3. 创建 if 条件: if (%val == 0)
  //   ReMapper.count(%val) -> 0 (false)
  //   三元运算符选择 ContextValue (即 %val)
  Value *Cond = Builder.CreateIsNull(
      ReMapper.count(ContextValue) ? ReMapper[ContextValue] : ContextValue);
  // IR 变化: 一条新指令被插入 "work" 块
  // %cond = icmp eq i32 %val, 0

  // 4. 【CFG 手术】拆分基本块
  Instruction *ThenTerm = nullptr, *ElseTerm = nullptr;
  SplitBlockAndInsertIfThenElse(Cond, &*BBHead, &ThenTerm, &ElseTerm);
```



**【`SplitBlockAndInsertIfThenElse` 后的 IR 状态 (中间态)】** `@foo` 函数的 CFG 被彻底改变：

```
entry:
  %val = add i32 %arg, 1
  br label %work ; (稍后 "work" 会被重命名)

work: ; (原始 "work" 块被拆分)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %clone.then, label %clone.else ; 新的终结符

clone.then: ; (新创建的空块)
  br label %tail ; ThenTerm 指向这条指令

clone.else: ; (新创建的空块)
  br label %tail ; ElseTerm 指向这条指令

tail: ; (新创建的块, 包含 "work" 的原始指令)
  ; BBHead (%mul) 和它之后的所有指令都在这里
  %mul = mul i32 %val, 2
  call void @print(i32 %mul)
  %add = add i32 %mul, 10
  ret i32 %add
```



**`cloneBB` (继续)**

```c++
  // 5. 获取 tail 块的指针
  BasicBlock *Tail = ThenTerm->getSuccessor(0); // Tail 指向 "tail" 块

  // 6. (断言检查)
  assert(Tail == ElseTerm->getSuccessor(0) && "Inconsistent CFG"); // 通过

  // 7. 重命名新块 (假设 DuplicateBBCount = 0)
  std::string DuplicatedBBId = std::to_string(DuplicateBBCount); // "0"
  ThenTerm->getParent()->setName("lt-clone-1-0"); // "clone.then"
  ElseTerm->getParent()->setName("lt-clone-2-0"); // "clone.else"
  Tail->setName("lt-tail-0");                     // "tail"
  // "work" 块被重命名
  ThenTerm->getParent()->getSinglePredecessor()->setName("lt-if-then-else-0");

  // 8. 准备局部 VMap
  ValueToValueMapTy TailVMap, ThenVMap, ElseVMap; // 均为空
  SmallVector<Instruction *, 8> ToRemove; // 为空

  // 9. 【核心克隆循环】遍历 "lt-tail-0" 中的所有指令
  for (auto IIT = Tail->begin(), IE = Tail->end(); IIT != IE; ++IIT) {
    
    // --- 循环 1: Instr = %mul = mul i32 %val, 2 ---
    Instruction &Instr = *IIT; // Instr 是 %mul
    if (Instr.isTerminator()) // false
    // 10. 克隆指令
    Instruction *ThenClone = Instr.clone(); // %mul.clone1 = mul i32 %val, 2
    Instruction *ElseClone = Instr.clone(); // %mul.clone2 = mul i32 %val, 2
    // 11. 修复克隆体1
    RemapInstruction(ThenClone, ThenVMap, ...); // ThenVMap 为空, 无变化
    ThenClone->insertBefore(ThenTerm->getIterator()); // 移入 "lt-clone-1-0"
    ThenVMap[&Instr] = ThenClone; // ThenVMap = { %mul -> %mul.clone1 }
    // 12. 修复克隆体2
    RemapInstruction(ElseClone, ElseVMap, ...); // ElseVMap 为空, 无变化
    ElseClone->insertBefore(ElseTerm->getIterator()); // 移入 "lt-clone-2-0"
    ElseVMap[&Instr] = ElseClone; // ElseVMap = { %mul -> %mul.clone2 }
    // 13. 检查是否为 void
    if (ThenClone->getType()->isVoidTy()) // false
    // 14. 创建 PHI 节点
    PHINode *Phi = PHINode::Create(ThenClone->getType(), 2); // %mul.phi = phi i32 ...
    Phi->addIncoming(ThenClone, ThenTerm->getParent()); // [ %mul.clone1, %lt-clone-1-0 ]
    Phi->addIncoming(ElseClone, ElseTerm->getParent()); // [ %mul.clone2, %lt-clone-2-0 ]
    // 15. 记录映射
    TailVMap[&Instr] = Phi; // TailVMap = { %mul -> %mul.phi }
    ReMapper[&Instr] = Phi; // ReMapper = { %mul -> %mul.phi }
    // 16. 在 "lt-tail-0" 中用 PHI 替换原始指令
    ReplaceInstWithInst(Tail, IIT, Phi);
    // "lt-tail-0" 中的 %mul 指令被 %mul.phi 替换

    // --- 循环 2: Instr = call void @print(i32 %mul) ---
    Instruction &Instr = *IIT; // Instr 是 @print
    if (Instr.isTerminator()) // false
    // 10. 克隆
    Instruction *ThenClone = Instr.clone(); // call void @print(i32 %mul)
    Instruction *ElseClone = Instr.clone(); // call void @print(i32 %mul)
    // 11. 修复克隆体1
    RemapInstruction(ThenClone, ThenVMap, ...);
    // 查找操作数 %mul。在 ThenVMap 中找到 -> %mul.clone1
    // ThenClone 被修复为: call void @print(i32 %mul.clone1)
    ThenClone->insertBefore(ThenTerm->getIterator()); // 移入 "lt-clone-1-0"
    ThenVMap[&Instr] = ThenClone;
    // 12. 修复克隆体2
    RemapInstruction(ElseClone, ElseVMap, ...);
    // 查找操作数 %mul。在 ElseVMap 中找到 -> %mul.clone2
    // ElseClone 被修复为: call void @print(i32 %mul.clone2)
    ElseClone->insertBefore(ElseTerm->getIterator()); // 移入 "lt-clone-2-0"
    ElseVMap[&Instr] = ElseClone;
    // 13. 检查是否为 void
    if (ThenClone->getType()->isVoidTy()) { // true
      // 13a. 将原始指令加入移除列表
      ToRemove.push_back(&Instr); // ToRemove = { &@print }
      continue; // 跳到下一轮循环
    }

    // --- 循环 3: Instr = %add = add i32 %mul, 10 ---
    Instruction &Instr = *IIT; // Instr 是 %add
    if (Instr.isTerminator()) // false
    // 10. 克隆
    Instruction *ThenClone = Instr.clone(); // %add.clone1 = add i32 %mul, 10
    Instruction *ElseClone = Instr.clone(); // %add.clone2 = add i32 %mul, 10
    // 11. 修复克隆体1
    RemapInstruction(ThenClone, ThenVMap, ...);
    // 查找操作数 %mul。在 ThenVMap 中找到 -> %mul.clone1
    // ThenClone 被修复为: %add.clone1 = add i32 %mul.clone1, 10
    ThenClone->insertBefore(ThenTerm->getIterator());
    ThenVMap[&Instr] = ThenClone; // ThenVMap = { %mul->%mul.1, %add->%add.1 }
    // 12. 修复克隆体2
    RemapInstruction(ElseClone, ElseVMap, ...);
    // 查找操作数 %mul。在 ElseVMap 中找到 -> %mul.clone2
    // ElseClone 被修复为: %add.clone2 = add i32 %mul.clone2, 10
    ElseClone->insertBefore(ElseTerm->getIterator());
    ElseVMap[&Instr] = ElseClone; // ElseVMap = { %mul->%mul.2, %add->%add.2 }
    // 13. 检查是否为 void
    if (ThenClone->getType()->isVoidTy()) // false
    // 14. 创建 PHI
    PHINode *Phi = PHINode::Create(ThenClone->getType(), 2); // %add.phi = phi i32 ...
    Phi->addIncoming(ThenClone, ThenTerm->getParent()); // [ %add.clone1, %lt-clone-1-0 ]
    Phi->addIncoming(ElseClone, ElseTerm->getParent()); // [ %add.clone2, %lt-clone-2-0 ]
    // 15. 记录映射
    TailVMap[&Instr] = Phi; // TailVMap = { %mul->%mul.phi, %add->%add.phi }
    ReMapper[&Instr] = Phi; // ReMapper = { %mul->%mul.phi, %add->%add.phi }
    // 16. 替换
    ReplaceInstWithInst(Tail, IIT, Phi);
    // "lt-tail-0" 中的 %add 指令被 %add.phi 替换

    // --- 循环 4: Instr = ret i32 %add ---
    Instruction &Instr = *IIT; // Instr 是 ret
    // 17. 检查是否为终结符
    if (Instr.isTerminator()) { // true
      // 17a. 修复终结符的操作数
      RemapInstruction(&Instr, TailVMap, RF_IgnoreMissingLocals);
      // 查找操作数 %add。在 TailVMap 中找到 -> %add.phi
      // Instr 被修复为: ret i32 %add.phi
      continue;
    }
  } // 9. for 循环结束

  // 18. 【清理】删除 "lt-tail-0" 中的 void 指令
  for (auto *I : ToRemove) // 遍历 ToRemove = { &@print }
    I->eraseFromParent(); // 原始的 "call void @print" 指令被删除

  // 19. 更新统计
  ++DuplicateBBCount; // DuplicateBBCount = 1
}
```

------



**最终结果：`@foo` 函数 (转换后)**

`cloneBB` 和 `run` 返回后，`@foo` 函数的 IR 变成了这样：

```
define i32 @foo(i32 %arg) {
entry:
  %val = add i32 %arg, 1
  br label %lt-if-then-else-0 ; (跳转到 "if" 块)

lt-if-then-else-0: ; (原 "work" 块的头部)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %lt-clone-1-0, label %lt-clone-2-0

lt-clone-1-0: ; (克隆体1)
  %mul.clone1 = mul i32 %val, 2
  call void @print(i32 %mul.clone1) ; 克隆的 void 指令
  %add.clone1 = add i32 %mul.clone1, 10
  br label %lt-tail-0 ; 原始终结符

lt-clone-2-0: ; (克隆体2)
  %mul.clone2 = mul i32 %val, 2
  call void @print(i32 %mul.clone2) ; 克隆的 void 指令
  %add.clone2 = add i32 %mul.clone2, 10
  br label %lt-tail-0 ; 原始终结符

lt-tail-0: ; (合并块)
  ; 原始指令被 PHI 节点替换
  %mul.phi = phi i32 [ %mul.clone1, %lt-clone-1-0 ], [ %mul.clone2, %lt-clone-2-0 ]
  ; 原始的 "call" 指令已被删除
  %add.phi = phi i32 [ %add.clone1, %lt-clone-1-0 ], [ %add.clone2, %lt-clone-2-0 ]
  ; 终结符的操作数已被修复
  ret i32 %add.phi
}
```
