# InjectFuncCall

## InjectFuncCall Pass使用 —— Transformation 转型

与`HelloWorld Pass`不同，此Pass**修改**程序的 LLVM IR，为程序添加新的功能

在程序的每个函数（Function）的**入口处**，插入一个如下 `printf` 函数的调用:

`printf("(llvm-tutor) Hello from: %s\n(llvm-tutor)   number of arguments: %d\n", FuncName, FuncNumArgs)`

这意味着经过这个 Pass 处理后的程序，在**运行时**，每当一个函数被**调用**，它就会立刻打印出自己的名字和参数个数

不同于HelloWorld在编译时进行opt操作即完成打印；

该Pass运行 `opt` 时**没有输出**，而是在原有IR代码中进行注入`print`，先生成新程序后，然后**运行新程序**才能看到输出



生成 Bitcode 文件: 即.bc文件

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -O0 -emit-llvm -c ../inputs/input_for_hello.c -o input_for_hello.bc
```

`-O0`（无优化）来确保 `clang` 不会优化掉任何函数，从而该 Pass 能处理所有函数



注入`print`调用

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libInjectFuncCall.dylib --passes="inject-func-call" input_for_hello.bc -o instrumented.bin
```

**`-o instrumented.bin`**: 这是**关键区别**！

没有使用 `-disable-output`，而是用 `-o` 将 `opt` **修改后**的 IR 保存到一个名为 `instrumented.bin` 的新文件中



用 LLVM 的解释器 `lli` 来运行它:

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/lli instrumented.bin
(llvm-tutor) Hello from: main
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
(llvm-tutor) Hello from: bar
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
(llvm-tutor) Hello from: fez
(llvm-tutor)   number of arguments: 3
(llvm-tutor) Hello from: bar
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
```

