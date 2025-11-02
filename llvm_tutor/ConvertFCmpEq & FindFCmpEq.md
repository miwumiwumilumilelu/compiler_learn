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

