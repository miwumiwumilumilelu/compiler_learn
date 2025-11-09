# MergeBB 

## MergeBB Pass使用 —— 合并相同基本块

**MergeBB** 会合并符合条件的相同基本块。在某种程度上，此过程会撤销 **DuplicateBB** 引入的转换。如下图所示：

```c++
BEFORE:                     AFTER DuplicateBB:                 AFTER MergeBB:
-------                     ------------------                 --------------
                              [ if-then-else ]                 [ if-then-else* ]
             DuplicateBB           /  \               MergeBB         |
[ BB ]      ------------>   [clone 1] [clone 2]      -------->    [ clone ]
                                   \  /                               |
                                 [ tail ]                         [ tail* ]

LEGEND:
-------
[BB]           - the original basic block
[if-then-else] - a new basic block that contains the if-then-else statement (**DuplicateBB**)
[clone 1|2]    - two new basic blocks that are clones of BB (**DuplicateBB**)
[tail]         - the new basic block that merges [clone 1] and [clone 2] (**DuplicateBB**)
[clone]        - [clone 1] and [clone 2] after merging, this block should be very similar to [BB] (**MergeBB**)
[label*]       - [label] after being updated by **MergeBB**
```

DuplicateBB 会将所有符合条件的基本块替换为四个新的基本块，其中两个是原始块的克隆。MergeBB 会将这两个克隆块合并在一起，但**不会删除 DuplicateBB 添加的剩余两个块（但会更新它们）**



**Run the Pass**

取以下IR内容为输入

```
llvm-tutor/build on  main [?] via 🅒 base 
➜ vim foo.ll                     

llvm-tutor/build on  main [?] via 🅒 base took 2.9s 
➜ cat foo.ll                     
define i32 @foo(i32) {
  %2 = icmp eq i32 %0, 19
  br i1 %2, label %3, label %5

; <label>:3:
  %4 = add i32 %0,  13
  br label %7

; <label>:5:
  %6 = add i32 %0,  13
  br label %7

; <label>:7:
  %8 = phi i32 [ %4, %3 ], [ %6, %5 ]
  ret i32 %8
}
```

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libMergeBB.dylib -passes="merge-bb" -S foo.ll -o merge.ll 

llvm-tutor/build on  main [?] via 🅒 base 
➜ cat merge.ll
; ModuleID = 'foo.ll'
source_filename = "foo.ll"

define i32 @foo(i32 %0) {
  %2 = icmp eq i32 %0, 19
  br i1 %2, label %3, label %3

3:                                                ; preds = %1, %1
  %4 = add i32 %0, 13
  br label %5

5:                                                ; preds = %3
  ret i32 %4
}
```

经处理后，输入模块中的基本块 3 和 5 已合并为一个基本块

再用DuplicateBB Pass的输出运用在该Pass上:

```sh
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -emit-llvm -S -O1 ../inputs/input_for_duplicate_bb.c -o input_for_duplicate_bb.ll 
```

现在我们将按顺序对 `foo` 应用 **DuplicateBB** 和 **MergeBB** 。请记住， **DuplicateBB** 需要 **RIV** ，这意味着我们总共需要加载三个插件：

```sh
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libRIV.dylib -load-pass-plugin ./lib/libMergeBB.dylib -load-pass-plugin ./lib/libDuplicateBB.dylib -passes=duplicate-bb,merge-bb -S input_for_duplicate_bb.ll -o merge_after_duplicate.ll
```

cat .ll:

```
define noundef i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
lt-if-then-else-0:
  %1 = icmp eq i32 %0, 0
  br i1 %1, label %lt-clone-2-0, label %lt-clone-2-0

lt-clone-2-0:                                     ; preds = %lt-if-then-else-0, %lt-if-then-else-0
  br label %lt-tail-0

lt-tail-0:                                        ; preds = %lt-clone-2-0
  ret i32 1
}
```

只有其中一个克隆 `lt-clone-2-0` 被保留了下来，并且 `lt-if-then-else-0` 已相应更新。无论 `if` 条件（更准确地说，是变量 `%1` ）的值如何，控制流都会跳转到 `lt-clone-2-0` 



## MergeBB 源码

### .h

```c++
bool canRemoveInst(const llvm::Instruction *Inst);
```

这是一个**安全检查**函数。`canMergeInstructions` 会调用它。如果一条指令有一次使用（`hasOneUse()`），合并它可能是危险的。此函数检查这次使用是否“安全”

* 其用户是后继块中的 `PHINode`（这是 `DuplicateBB` 克隆体合并时的标准情况）

* 其用户和它在同一个块中（如果块被删除，用户也会一起被删除，所以是安全的

```c++
bool canMergeInstructions(llvm::ArrayRef<llvm::Instruction *> Insts);
```

这是一个**检查**函数。它接受一对指令（`Insts`），并判断它们是否“相同”到可以合并。这不仅包括操作码相同，还包括操作数也必须完全相同

```c++
unsigned updateBranchTargets(llvm::BasicBlock *BBToErase,
                             llvm::BasicBlock *BBToRetain);
```

这是一个**执行**函数。当 `mergeDuplicatedBlock` 确定 `BBToErase` (要删除的块) 和 `BBToRetain` (要保留的块) 可以合并时，此函数负责修改控制流图（CFG）。它会找到所有跳转到 `BBToErase` 的前驱块，并将它们的跳转目标重定向到 `BBToRetain`

```c++
bool
mergeDuplicatedBlock(llvm::BasicBlock *BB,
                     llvm::SmallPtrSet<llvm::BasicBlock *, 8> &DeleteList);
```

这是**核心逻辑**函数。它接受一个基本块 `BB`，然后尝试在函数中为 `BB` 寻找到一个“孪生兄弟”（内容完全相同的另一个块）。如果找到了，它就执行合并，并将 `BB` 添加到 `DeleteList` (待删除列表) 中

```c++
class LockstepReverseIterator {
  llvm::BasicBlock *BB1;
  llvm::BasicBlock *BB2;

  llvm::SmallVector<llvm::Instruction *, 2> Insts;
  bool Fail;

public:
  LockstepReverseIterator(llvm::BasicBlock *BB1In, llvm::BasicBlock *BB2In);

  llvm::Instruction *getLastNonDbgInst(llvm::BasicBlock *BB);
  bool isValid() const { return !Fail; }

  void operator--();

  llvm::ArrayRef<llvm::Instruction *> operator*() const { return Insts; }
};
```

这是一个自定义的迭代器类，也是这个 Pass 得以实现的关键工具

- **目的**：为了比较 `BB1` 和 `BB2` 是否相同，你需要逐条指令地比较它们。这个类允许你“同步地”（in lockstep）从 `BB1` 和 `BB2` 的末尾向前反向迭代**（逆序遍历，后向分析）**

- **`LockstepReverseIterator(BB1, BB2)`** (构造函数): 设置迭代器，使其指向 `BB1` 和 `BB2` 的最后一个非调试指令（即终结符之前的最后一条“真实”指令）。

- **`isValid()`**: 检查迭代是否完成。如果已经到达了任一基本块的开头，则返回 `false`。

- **`operator--()`** (递减): **同时**将 `BB1` 和 `BB2` 的内部指针移动到它们的“上一条”非调试指令。

- **`operator*()`** (解引用): 返回一个包含**当前这对指令**的数组（`[Inst_from_BB1, Inst_from_BB2]`）。`canMergeInstructions` 就会接收这个数组。

  `llvm::ArrayRef<llvm::Instruction *> operator*() const { return Insts; }`

  其中operator* ()是重载符

  这使得你可以对 `LockstepReverseIterator` 类的对象（比如 `LRI`）使用 `*` 符号，就像它是一个 C++ 的标准指针或迭代器一样

  当写下 `*LRI` 时，C++ 编译器会自动将其翻译为 `LRI.operator*()`

  当 `operator*` 被调用时，它会返回一个指向 `Instruction*` 数组的轻量级视图——**`ArrayRef<llvm::Instruction *>`**

  这个 `const` 关键字放在函数末尾，意味着这个函数是一个**“只读”**操作。



### .cpp

**run**

```c++
PreservedAnalyses MergeBB::run(llvm::Function &Func,
                               llvm::FunctionAnalysisManager &) {
  bool Changed = false;
  SmallPtrSet<BasicBlock *, 8> DeleteList;
  for (auto &BB : Func) {
    Changed |= mergeDuplicatedBlock(&BB, DeleteList);
  }

  for (BasicBlock *BB : DeleteList) {
    DeleteDeadBlock(BB);
  }

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
```

`SmallPtrSet<BasicBlock *, 8> DeleteList;`创建了一个8个BasicBlock *指针大小的 `DeleteList`。这是一个为指针优化的**哈希集合 (Set)**，用于跟踪所有“已合并”并等待删除的基本块

遍历函数中的所有基本块，并将其（`&BB`）作为 `BB1` 传递给 `mergeDuplicatedBlock`。`mergeDuplicatedBlock` 是真正的核心函数，它会尝试为 `BB1` 寻找一个“孪生兄弟” `BB2` 并进行合并

* 如果 `mergeDuplicatedBlock` 成功了（返回 `true`），`BB1` 就会被添加到 `DeleteList` 中，并且 `Changed` 标志位被设为 `true`

设置标志位用来告诉管理器是否修改了IR



**mergeDuplicatedBlock**

**入参资格预审：**

1. BB1不能是入口块

   入口块 (`entry`) 是函数的起点，不能被合并掉

2. 必须以“无条件分支”结束

   大大简化了分析，因为它保证了 `BB1` **只有一个**确定的后继块（`BBSucc`）

3. 它的所有前驱块必须是 'br' 或 'switch'

   为了确保 `updateBranchTargets` 函数（稍后执行合并）可以轻松地重定向它们

```c++
bool MergeBB::mergeDuplicatedBlock(BasicBlock *BB1,
                                   SmallPtrSet<BasicBlock *, 8> &DeleteList) {
  // Do not optimize the entry block
  if (BB1 == &BB1->getParent()->getEntryBlock())
    return false;

  // Only merge CFG edges of unconditional branch
  BranchInst *BB1Term = dyn_cast<BranchInst>(BB1->getTerminator());
  if (!(BB1Term && BB1Term->isUnconditional()))
    return false;

  // Do not optimize non-branch and non-switch CFG edges (to keep things
  // relatively simple)
  for (auto *B : predecessors(BB1))
    if (!(isa<BranchInst>(B->getTerminator()) ||
          isa<SwitchInst>(B->getTerminator())))
      return false;
```

* `BB1->getParent()`：一个 `BasicBlock` 的“父亲”（Parent）是包含它的那个 `Function`

​	`getEntryBlock()` 是 `Function` 类的一个成员函数，这个函数返回的是一个 `llvm::BasicBlock &` 类型，即**对入口块的引用**

​	最后进行取地址操作&

* `BB1->getTerminator()`：获取BB1基本块的终结符指令

* `!(BB1Term && BB1Term->isUnconditional())`

  如果 `BB1Term` 是 `nullptr`（即终结符不是 `BranchInst`），这个 `&&` 表达式的第一部分就是 `false`

  `isUnconditional()` 是 `BranchInst` 的一个成员函数，它检查这个分支是无条件的 (`br label %dest`) 还是有条件的 (`br i1 %cond, ...`)

* `predecessors(BB1)`是一个 LLVM 的辅助函数，它会返回一个**可迭代的列表**，这个列表包含了**所有能够跳转到 `BB1` 的前驱基本块**

  直接或间接需要包含这个头文件`#include "llvm/IR/CFG.h"`

* `!(isa<BranchInst>(B->getTerminator()) ||  isa<SwitchInst>(B->getTerminator()))`

  既不是BranchInst分支指令，也不是SwitchInst

  `isa<>` 是 LLVM 的“is-a”类型检查

**分析后继块 (BBSucc)**

如果两个块是相同的，它们很可能会跳转到同一个后继块

该板块负责锁定这个共同的后继块

```c++
  BasicBlock *BBSucc = BB1Term->getSuccessor(0);

  BasicBlock::iterator II = BBSucc->begin();
  const PHINode *PN = dyn_cast<PHINode>(II);
  Value *InValBB1 = nullptr;
  Instruction *InInstBB1 = nullptr;
  BBSucc->getFirstNonPHI();
  if (nullptr != PN) {
    // Do not optimize if multiple PHI instructions exist in the successor (to
    // keep things relatively simple)
    if (++II != BBSucc->end() && isa<PHINode>(II))
      return false;

    InValBB1 = PN->getIncomingValueForBlock(BB1);
    InInstBB1 = dyn_cast<Instruction>(InValBB1);
  }
```

`BasicBlock *BBSucc = BB1Term->getSuccessor(0);`

获取BB1的唯一后继块（因为之前已经通过检查终结符指令为br无条件跳转，说明了BB1只有一个后继块）

设置迭代器起点II

检查后继块第一条指令是否是Phi指令`if (nullptr != PN)`

`if (++II != BBSucc->end() && isa<PHINode>(II))`如果后面的非终结符指令仍然是Phi指令，即有多个Phi指令，则直接不进行处理

即此 Pass 只处理 0 或 1 个 PHI 节点的情况

暂存BB1传入Phi节点的值`InValBB1`，并判断这个传入的Value是否是指令`dyn_cast<Instruction>(InValBB1);`

**搜索循环与“候选者” (BB2) 的快速过滤**

```c++
  unsigned BB1NumInst = getNumNonDbgInstrInBB(BB1);
  for (auto *BB2 : predecessors(BBSucc)) {
    // Do not optimize the entry block
    if (BB2 == &BB2->getParent()->getEntryBlock())
      continue;

    // Only merge CFG edges of unconditional branch
    BranchInst *BB2Term = dyn_cast<BranchInst>(BB2->getTerminator());
    if (!(BB2Term && BB2Term->isUnconditional()))
      continue;

    // Do not optimize non-branch and non-switch CFG edges (to keep things
    // relatively simple)
    for (auto *B : predecessors(BB2))
      if (!(isa<BranchInst>(B->getTerminator()) ||
            isa<SwitchInst>(B->getTerminator())))
        continue;

    // Skip basic blocks that have already been marked for merging
    if (DeleteList.end() != DeleteList.find(BB2))
      continue;

    // Make sure that BB2 != BB1
    if (BB2 == BB1)
      continue;

    // BB1 and BB2 are definitely different if the number of instructions is
    // not identical
    if (BB1NumInst != getNumNonDbgInstrInBB(BB2))
      continue;
```

获取BB1基本块中的非调试指令数量`BB1NumInst`

遍历BB1选定的唯一后继块的所有前驱基本块，来判断是否是BB2

首先检查BB2：

1. 需要不是入口块

2. 需要终结符指令是br无条件跳转指令，只有唯一后继块

3. 需要其所有前驱块的终结符指令，必须是br或者switch，方便合并时定向

4. 需要保证不在待删除队列中`DeleteList.end() != DeleteList.find(BB2)`

   **`DeleteList.find(BB2)`**

   - 这个函数会在 `DeleteList` 集合中**搜索** `BB2`
   - **如果找到了**：它会返回一个**迭代器**，指向 `BB2` 在集合中的位置
   - **如果没找到**：它会返回一个特殊的“哨兵”迭代器，这个哨兵就是 `DeleteList.end()`，不是指向集合的最后一个元素，用来表示“结束”或“未找到”

5. BB2 ! = BB1

6. 其指令数量必须等于BB2中指令数量（最基本）

**PHI 节点一致性检查 (关键逻辑)**

检查BB2遍历循环中，如果BB1的后继基本块中有Phi节点，那么就检查BB2的后继基本块是否有Phi节点且一致

```c++
    if (nullptr != PN) {
      Value *InValBB2 = PN->getIncomingValueForBlock(BB2);
      Instruction *InInstBB2 = dyn_cast<Instruction>(InValBB2);

      bool areValuesSimilar = (InValBB1 == InValBB2);
      bool bothValuesDefinedInParent =
          ((InInstBB1 && InInstBB1->getParent() == BB1) &&
           (InInstBB2 && InInstBB2->getParent() == BB2));
      if (!areValuesSimilar && !bothValuesDefinedInParent)
        continue;
    }
```

**`areValuesSimilar` (简单情况)** :

 `BB1` 和 `BB2` 为 PHI 节点提供了**完全相同的值**。例如，它们都传入常量 `0`，或者都传入在它们之前定义的某个变量 `%x`。这是安全的

**`bothValuesDefinedInParent` (复杂情况)** :

`BB1` 传入 `%v1`，`BB2` 传入 `%v2`。这两个值**不同**，但是 `%v1` 是在 `BB1` *内部*定义的，而 `%v2` 是在 `BB2` *内部*定义的。如果 `BB1` 和 `BB2` 真是“孪生兄弟”，那么定义 `%v1` 和 `%v2` 的指令也应该是相同的

如：

`%v2 = add i32 %x, 10` 

`%v1 = add i32 %x, 10`

**深度比较：逐条指令验证**

```c++
    // Finally, check that all instructions in BB1 and BB2 are identical
    LockstepReverseIterator LRI(BB1, BB2);
    while (LRI.isValid() && canMergeInstructions(*LRI)) {
      --LRI;
    }

    // Valid iterator  means that a mismatch was found in middle of BB
    if (LRI.isValid())
      continue;
```

`LockstepReverseIterator` 被创建，它会跳过终结符和调试指令，指向 `BB1` 和 `BB2` 的**最后一条真实指令**

从后向前遍历每条真实指令`--LRI`

1. `LRI.isValid()`: 检查是否已到达块的开头
2. `canMergeInstructions(*LRI)`: 调用辅助函数（在 `.h` 中定义）来比较这对指令是否**完全相同**（相同的操作码，相同的操作数，并且使用安全）

进行失败判断：

如果 `while` 循环因为 `canMergeInstructions` 返回 `false` 而**中途退出**，此时 `LRI.isValid()` **仍然为 true**。这说明在未遍历完该基本块之前找到了一条不匹配的指令，因此 `continue` 到下一个 `BB2` 候选者

**执行合并与收尾**

```c++
    unsigned UpdatedTargets = updateBranchTargets(BB1, BB2);
    assert(UpdatedTargets && "No branch target was updated");
    OverallNumOfUpdatedBranchTargets += UpdatedTargets;
    DeleteList.insert(BB1);
    NumDedupBBs++;

    return true;
  }

  return false;
}
```

`updateBranchTargets`会找到所有跳转到 `BB1` 的前驱块，并将它们的跳转目标（`br` 或 `switch`）**重定向到 `BB2`**。`BB1` 现在成了“死代码”基本块

`OverallNumOfUpdatedBranchTargets` 是文件顶部用 `STATISTIC` 宏定义的**全局统计变量**。

- 它将刚刚更新的跳转目标数量 (`UpdatedTargets`)，**累加**到全局的 `OverallNumOfUpdatedBranchTargets` 计数器中

- 这是为了给 LLVM 的 `-stats` 功能提供数据
- 当 Pass 运行完毕后，可以通过 `opt` 的 `-stats` 选项查看 Pass 的运行报告
- 这一行代码会告诉你，`MergeBB` Pass 在**整个**函数中总共修改了**多少条**终结符指令（`br` 或 `switch`）

`NumDedupBBs`也是一个用 `STATISTIC` 宏定义的**全局统计变量**。

- `++` (自增) 操作符将 `NumDedupBBs` 计数器加 1

- 同样是为了 `-stats` 报告
- 这一行代码在每一次成功的合并（`BB1` 被合并到 `BB2`）时执行一次
- 运行结束后，这个统计数据将告诉你 `MergeBB` Pass 总共**合并/删除**了多少个基本块



**getNumNonDbgInstrInBB**

```c++
static unsigned getNumNonDbgInstrInBB(BasicBlock *BB) {
  unsigned Count = 0;
  for (Instruction &Instr : *BB)
    if (!isa<DbgInfoIntrinsic>(Instr))
      Count++;
  return Count;
}
```

获得真实指令数量，遍历基本块中的指令，如果不是调试指令，则count++



**canMergeInstructions**

```c++
bool MergeBB::canMergeInstructions(ArrayRef<Instruction *> Insts) {
  const Instruction *Inst1 = Insts[0];
  const Instruction *Inst2 = Insts[1];
  
  if (!Inst1->isSameOperationAs(Inst2))
    return false;

  bool HasUse = !Inst1->user_empty();
  for (auto *I : Insts) {
    if (HasUse && !I->hasOneUse())
      return false;
    if (!HasUse && !I->user_empty())
      return false;
  }
  
  if (HasUse) {
    if (!canRemoveInst(Inst1) || !canRemoveInst(Inst2))
      return false;
  }

  assert(Inst2->getNumOperands() == Inst1->getNumOperands());
  auto NumOpnds = Inst1->getNumOperands();
  for (unsigned OpndIdx = 0; OpndIdx != NumOpnds; ++OpndIdx) {
    if (Inst2->getOperand(OpndIdx) != Inst1->getOperand(OpndIdx))
      return false;
  }
  return true;
}
```

**入参`ArrayRef<Instruction *> Insts`是`LockstepReverseIterator`类的解引用**

首先对指令组合进行如下检查：

1. 验证两条指令是否具有相同的操作码`Inst1->isSameOperationAs(Inst2)`

2. 检查是否是相同数量的Use

   * Inst1有使用点，且二者如果存在任意一个使用点不止一个的情况，则不行`HasUse && !I->hasOneUse()`
   * Inst1没有使用点，但另一个即Inst2有使用点，则也不行`!HasUse && !I->user_empty()`

   即需要二者要么都0个Use，要么都只有1个Use

3. 如果二者都只有1个Use，要求确保这个一次使用是安全的

   * 即Use处要么在**同一个块**中（会一起被删除）
   * 要么是**后继块的 `PHINode`**（`DeleteDeadBlock` 知道如何修复）

4. 源操作数检查：

   首先操作数数量需要相同`Inst2->getNumOperands() == Inst1->getNumOperands()`

   且每个源操作数需要一一对应相等`Inst2->getOperand(OpndIdx) != Inst1->getOperand(OpndIdx)`

   这和`bothValuesDefinedInParent`并不冲突，`bothValuesDefinedInParent`是指目的操作数不同



**canRemoveInst**

如果删除了 `Inst`（及其所在的整个基本块），`Inst` 的**那个唯一的Use点** 会不会因此“损坏”并导致 IR (LLVM IR) 非法

```c++
bool MergeBB::canRemoveInst(const Instruction *Inst) {
  assert(Inst->hasOneUse() && "Inst needs to have exactly one use");

  auto *PNUse = dyn_cast<PHINode>(*Inst->user_begin());
  auto *Succ = Inst->getParent()->getTerminator()->getSuccessor(0);
  auto *User = cast<Instruction>(*Inst->user_begin());

  bool SameParentBB = (User->getParent() == Inst->getParent());
  bool UsedInPhi = (PNUse && PNUse->getParent() == Succ &&
                    PNUse->getIncomingValueForBlock(Inst->getParent()) == Inst);

  return UsedInPhi || SameParentBB;
}
```

`user_begin()`获取第一个用户（也是唯一的用户）

看能否转型成功，判断是不是Phi指令

获取后继基本块指针`*Succ`（获取 br 的第一个（也是唯一的）目标 (BBSucc)）

将这个用户（它是一个 Value* ）转换为 Instruction*

两种情况被允许，被认为是安全的：

1. Use和Def在同一基本块中`User->getParent() == Inst->getParent()`
2. 是Phi指令，Phi指令是当前基本块的下一个基本块中使用，且当前Inst作为Inst所在块的参数传入到了Phi指令



**updateBranchTargets**

找到所有跳转到 `BBToErase`（要删除的块）的“前驱块”，并将它们的目标**重定向**到 `BBToRetain`（要保留的块）

**`LLVM_DEBUG`**：这是一个调试宏。只有在 `opt` 命令（LLVM 优化器）使用 `-debug` 标志运行时，这行代码才会被编译并打印调试信息

```c++
unsigned MergeBB::updateBranchTargets(BasicBlock *BBToErase, BasicBlock *BBToRetain) {
  SmallVector<BasicBlock *, 8> BBToUpdate(predecessors(BBToErase));

  LLVM_DEBUG(dbgs() << "DEDUP BB: merging duplicated blocks ("
                    << BBToErase->getName() << " into " << BBToRetain->getName()
                    << ")\n");

  unsigned UpdatedTargetsCount = 0;
  for (BasicBlock *BB0 : BBToUpdate) {
    // The terminator is either a branch (conditional or unconditional) or a
    // switch statement. One of its targets should be BBToErase. Replace
    // that target with BBToRetain.
    Instruction *Term = BB0->getTerminator();
    for (unsigned OpIdx = 0, NumOpnds = Term->getNumOperands();
         OpIdx != NumOpnds; ++OpIdx) {
      if (Term->getOperand(OpIdx) == BBToErase) {
        Term->setOperand(OpIdx, BBToRetain);
        UpdatedTargetsCount++;
      }
    }
  }

  return UpdatedTargetsCount;
}
```

OpIdx为索引遍历前驱基本块的操作数，找到`Term->getOperand(OpIdx) == BBToErase`操作数名为BB1（本例需要删除的）的

将其重新设置为BB2`Term->setOperand(OpIdx, BBToRetain)`

更新计数
