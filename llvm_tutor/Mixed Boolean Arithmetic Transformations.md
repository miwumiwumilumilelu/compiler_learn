# Mixed Boolean Arithmetic Transformations

## Mixed Boolean Arithmetic Transformations Pass使用——分析 Sub&Add 的混淆替换公式

**MBASub :** 

`a - b == (a + ~b) + 1`

-b 等于 b 取非 + 1

上述公式替换了所有整数 `sub` 的实例，相应的 LIT 测试验证了公式和实现的正确性

使用 input_for_mba_sub.c测试 MBASub ：

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -emit-llvm -S ../inputs/input_for_mba_sub.c -o input_for_sub.ll
../inputs/input_for_mba_sub.c:10:10: fatal error: 'stdio.h' file not found
   10 | #include <stdio.h>
      |          ^~~~~~~~~
1 error generated.
```

自己编译的 `clang` (`~/projects/llvm-project/build/bin/clang`) 是一个“纯净”的编译器，它**不知道去哪里寻找操作系统macOS自带的标准库头文件**（ `stdio.h`）,macOS的系统根目录叫做SDK



**`-isysroot`**: 这是一个编译器标志，用来告诉 `clang`：请把后面给你的这个路径当作你的系统根目录，去那里寻找 `<stdio.h>` 这样的头文件

**`$(xcrun --show-sdk-path)`**: 这部分会在 `clang` 命令执行**之前**被您的 shell（zsh）先执行

- `xcrun --show-sdk-path` 会输出您当前 Xcode Command Line Tools 对应的 **macOS SDK 的完整路径**
- `$(...)` 语法会把这个输出的路径直接替换到命令行中

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -isysroot $(xcrun --show-sdk-path) -emit-llvm -S ../inputs/input_for_mba_sub.c -o input_for_sub.ll
```

opt操作

```shell
~/projects/llvm-project/build/bin/opt -load-pass-plugin=./lib/libMBASub.dylib -passes="mba-sub" -S input_for_sub.ll -o out.ll
```

lli操作

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/lli out.ll                                                                                 
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.	Program arguments: /Users/manbin/projects/llvm-project/build/bin/lli out.ll
 #0 0x000000010111a144 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/Users/manbin/projects/llvm-project/build/bin/lli+0x100bba144)
 #1 0x0000000101118080 llvm::sys::RunSignalHandlers() (/Users/manbin/projects/llvm-project/build/bin/lli+0x100bb8080)
 #2 0x000000010111abf4 SignalHandler(int, __siginfo*, void*) (/Users/manbin/projects/llvm-project/build/bin/lli+0x100bbabf4)
 #3 0x00000001870096a4 (/usr/lib/system/libsystem_platform.dylib+0x1804ad6a4)
 #4 0x0000000186e6cb00 (/usr/lib/system/libsystem_c.dylib+0x180310b00)
 #5 0x0000000186e6cb00 (/usr/lib/system/libsystem_c.dylib+0x180310b00)
 #6 0x0000000102ac0024
 #7 0x0000000100df4724 llvm::orc::runAsMain(int (*)(int, char**), llvm::ArrayRef<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>, std::__1::optional<llvm::StringRef>) (/Users/manbin/projects/llvm-project/build/bin/lli+0x100894724)
 #8 0x0000000100566744 runOrcJIT(char const*) (/Users/manbin/projects/llvm-project/build/bin/lli+0x100006744)
 #9 0x0000000100562330 main (/Users/manbin/projects/llvm-project/build/bin/lli+0x100002330)
#10 0x0000000186c2eb98
[1]    15822 segmentation fault  ~/projects/llvm-project/build/bin/lli out.ll
```

段错误segmentation fault：

`input_for_mba_sub.c` 中包含了 `#include <stdio.h>` 和 `#include <stdlib.h>`，并调用了 C 标准库里的函数，比如 `atoi`（将字符串转为整数）。

因此，生成的 `out.ll` 文件中包含了对外部函数 `@atoi` 的调用。

**`lli` 本身并不知道如何找到并执行 C 标准库里的函数。** 它就像一个只懂 LLVM IR 语言的翻译官，当它遇到一个它不认识的外部函数时，就不知道该怎么办了，最终导致程序崩溃



Cat 操作，直接看结果：

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ cat out.ll                  
; ModuleID = 'input_for_sub.ll'
source_filename = "../inputs/input_for_mba_sub.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define i32 @main(i32 noundef %0, ptr noundef %1) #0 {
		# ...省略
		%28 = load i32, ptr %6, align 4   ; 加载变量 a 的值到 %28
 		%29 = load i32, ptr %7, align 4   ; 加载变量 b 的值到 %29
		#整数减法e = a - b 替换成e = (a + ~b) + 1
		%30 = xor i32 %29, -1             ; 对 b (%29) 按位取反
 		%31 = add i32 %28, %30            ; 计算 a + ~b
 		%32 = add i32 %31, 1              ; 结果再加 1
 		store i32 %32, ptr %10, align 4   ; 将最终结果 (e) 存入内存
		# ...省略，还有一次减法转换，这里不再说明
}

declare i32 @atoi(ptr noundef) #1

attributes #0 = { noinline nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 15, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 21.1.3 (git@github.com:llvm/llvm-project.git c6af6be3cd1cbfa0dcd05ff9b8bda457a1902ba0)"}
```



**MBAAdd :**

MBAAdd 过程实现了稍微复杂的公式，该公式**仅对 8 位整数有效**：

`a + b == (((a ^ b) + 2 * (a & b)) * 39 + 23) * 151 + 111`

与 `MBASub` 类似，它根据上述恒等式替换所有整数 `add` 实例，但仅适用于 8 位整数。LIT 测试验证了公式和实现的正确性

使用input_for_mba.c测试 MBAAdd ：

```shell
~/projects/llvm-project/build/bin/clang -isysroot $(xcrun --show-sdk-path) -O1 -emit-llvm -S ../inputs/input_for_mba.c -o input_for_mba.ll
```

```shell
~/projects/llvm-project/build/bin/opt -load-pass-plugin=./lib/libMBAAdd.dylib -passes="mba-add" -S input_for_mba.ll -o out.ll
```



使用cat命令查看结果：

原始 C 代码中的 `foo` 函数处理的是 `int8_t`（8位整数），并且包含三次加法运算。在 `out.ll` 的 `@foo` 函数中，发现**已经没有任何 `add` 指令了**。它们全部被替换成了庞大而复杂的指令序列

`c + d` : (IR 中是 `%0` 和 `%1`)

```
; define noundef signext i8 @foo(i8 noundef signext %0, i8 noundef signext %1, ...)

; --- 对应 (a ^ b) + 2 * (a & b) ---
 %5 = xor i8 %1, %0              ; a ^ b
 %6 = and i8 %1, %0              ; a & b
 %7 = mul i8 2, %6               ; 2 * (a & b)
 %8 = add i8 %5, %7              ; (a ^ b) + 2 * (a & b)

; --- 对应 (... * 39 + 23) * 151 + 111 ---
 %9 = mul i8 39, %8              ; ... * 39
 %10 = add i8 23, %9             ; ... + 23
 %11 = mul i8 -105, %10          ; ... * 151 (注意：151 对于 8 位有符号整数是 -105)
 %12 = add i8 111, %11           ; ... + 111
```

`@foo` 函数被完全混淆了，这证明 `MBAAdd` Pass 成功地识别并转换了所有的 8 位整数加法

而在main中：

```
define ... i32 @main(...) ... {
  ; ... 省略了 atoi 和 load 指令 ...
  
  %15 = add i32 %8, %5
  %16 = add i32 %15, %11
  %17 = add i32 %16, %14

  ; ...
  ret i32 %19
}
```

`@main` 函数中的 **`add` 指令完好无损地保留了下来**！它们并没有被替换成复杂的公式

因为`@main` 函数中处理的变量是 `i32` 类型，而 `MBAAdd` Pass 被明确设计为**只处理 8 位整数**的加法



## Mixed Boolean Arithmetic Transformations 源码

### MBASub.h

PassInfoMixIn 是一个 CRTP 混合模块，用于自动提供通行证所需的信息性 API， 目前它仅提供“name”方法

```cpp
struct MBASub : public llvm::PassInfoMixin<MBASub>{...}
```

### MBASub.c

**核心函数：`runOnBasicBlock`**

向下转型`BinaryOperator`指针，检查当前指令是否是二元操作

`BinOp->getOpcode();`获得操作符，检查 二元操作指令操作符是不是减法 或者 二元操作指令类型是不是整型，以确保进行变换的是**二元整型减法操作**

`IRBuilder<> Builder(BinOp);`定位当前选到的指令，构建IRBuilder

```c++
Instruction *NewValue = BinaryOperator::CreateAdd(
        Builder.CreateAdd(BinOp->getOperand(0),       // a
                          Builder.CreateNot(BinOp->getOperand(1))), // ~b
        ConstantInt::get(BinOp->getType(), 1));        // 1

// `BinOp->getOperand(0)`获取指令的第一个操作数a，1则是b

// 此处也可以写`Instruction *NewValue = Builder.CreateAdd`

// `ConstantInt::get(Type, Value)`制造IR构建所需的同类型同值的常量
```

`ReplaceInstWithInst(&BB, Inst, NewValue);`两个Instruction型的取代

最后标记changed为true，即已修改已替换；并更新统计计数



**运行主函数：`run`**

```c++
  for (auto &BB : F) {
    Changed |= runOnBasicBlock(BB);
  }
  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
```

只要找到了一个符合条件的指令并修改了则Changed保持为1

并且如果修改了，返回的结果告知编译器“之前的分析结果都失效”



### MBAAdd.h

和Sub一样模板

```c++
struct MBAAdd : public llvm::PassInfoMixin<MBAAdd> 
```

### MBAAdd.c

`BinOp->getType()->getIntegerBitWidth() == 8`检查是否是8位，获取二元操作指令的操作数位数，只对8位整型进行替换

通过`ConstantInt::get`准备常量，为了构建IR

```c++
    // Constants used in building the instruction for substitution
    auto Val39 = ConstantInt::get(BinOp->getType(), 39);
    auto Val151 = ConstantInt::get(BinOp->getType(), 151);
    auto Val23 = ConstantInt::get(BinOp->getType(), 23);
    auto Val2 = ConstantInt::get(BinOp->getType(), 2);
    auto Val111 = ConstantInt::get(BinOp->getType(), 111);
```

```c++
    Instruction *NewInst =
        // E = e5 + 111
        BinaryOperator::CreateAdd(
            Val111,
            // e5 = e4 * 151
            Builder.CreateMul(
                Val151,
                // e4 = e2 + 23
                Builder.CreateAdd(
                    Val23,
                    // e3 = e2 * 39
                    Builder.CreateMul(
                        Val39,
                        // e2 = e0 + e1
                        Builder.CreateAdd(
                            // e0 = a ^ b
                            Builder.CreateXor(BinOp->getOperand(0),
                                              BinOp->getOperand(1)),
                            // e1 = 2 * (a & b)
                            Builder.CreateMul(
                                Val2, Builder.CreateAnd(BinOp->getOperand(0),
                                                        BinOp->getOperand(1))))
                    ) // e3 = e2 * 39
                ) // e4 = e2 + 23
            ) // e5 = e4 * 151
        ); // E = e5 + 111
```

其他地方同理Sub