# InjectFuncCall

## InjectFuncCall Pass使用 —— Transformation 转型

此过程是一个用于代码检测的 *HelloWorld* 示例。对于输入模块中定义的每个函数， **InjectFuncCall** 都会将以下调用添加（注入）到 printf中

`printf("(llvm-tutor) Hello from: %s\n(llvm-tutor)   number of arguments: %d\n", FuncName, FuncNumArgs)`

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -O0 -emit-llvm -c ../inputs/input_for_hello.c -o input_for_hello.bc
```

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libInjectFuncCall.dylib --passes="inject-func-call" input_for_hello.bc -o instrumented.bin
```

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

