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

入参资格预审：

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

  

