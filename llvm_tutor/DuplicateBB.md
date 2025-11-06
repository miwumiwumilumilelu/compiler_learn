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

### .cpp