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

### .cpp