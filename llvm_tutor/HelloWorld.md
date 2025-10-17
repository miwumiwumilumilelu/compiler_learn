# HelloWorld Pass

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ cmake -DLT_LLVM_INSTALL_DIR=~/projects/llvm-project/build ..
```

这个命令**并没有**将 `llvm-tutor` 的代码“插入”到 `llvm-project` 中。它的实际作用是：在 `llvm-tutor` 的 `build` 目录中，**配置一个独立的编译环境**，并告诉这个环境：“编译 `llvm-tutor` 所需要的所有依赖项（比如 LLVM 的头文件和库），请到 `-DLT_LLVM_INSTALL_DIR` 这个路径（也就是 `~/projects/llvm-project/build`）去找



```shell
llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ ls
cmake_install.cmake CMakeFiles          Makefile

llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ make                                                         
[100%] Built target HelloWorld
```

在build环境中，不污染llvm-tutor源代码

编译后得到的`.dylib` 是 **动态链接库 (Dynamic Link Library)** 文件在苹果的 macOS 上的文件扩展名

它本身不能独立运行，但可以被 `opt` 这样的主程序加载，为主程序提供额外的功能（也就是 `HelloWorld` Pass)

```shell
llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ ls -l ../lib/
total 7216
-rw-r--r--   1 manbin  staff    1389 10 17 20:52 cmake_install.cmake
drwxr-xr-x  15 manbin  staff     480 10 17 20:52 CMakeFiles
-rwxr-xr-x   1 manbin  staff  254696 10 17 20:52 libConvertFCmpEq.dylib
-rwxr-xr-x   1 manbin  staff  436328 10 17 20:52 libDuplicateBB.dylib
-rwxr-xr-x   1 manbin  staff  300248 10 17 20:52 libDynamicCallCounter.dylib
-rwxr-xr-x   1 manbin  staff  342480 10 17 20:52 libFindFCmpEq.dylib
-rwxr-xr-x   1 manbin  staff  167168 10 17 20:52 libHelloWorld.dylib
-rwxr-xr-x   1 manbin  staff  251992 10 17 20:52 libInjectFuncCall.dylib
-rwxr-xr-x   1 manbin  staff  218896 10 17 20:52 libMBAAdd.dylib
-rwxr-xr-x   1 manbin  staff  216416 10 17 20:52 libMBASub.dylib
-rwxr-xr-x   1 manbin  staff  299408 10 17 20:52 libMergeBB.dylib
-rwxr-xr-x   1 manbin  staff  373944 10 17 20:52 libOpcodeCounter.dylib
-rwxr-xr-x   1 manbin  staff  416032 10 17 20:52 libRIV.dylib
-rwxr-xr-x   1 manbin  staff  363160 10 17 20:52 libStaticCallCounter.dylib
-rw-r--r--   1 manbin  staff   21802 10 17 20:52 Makefile
```



clang编译.c文件得到中间IR

```shell
llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -O1 -S -emit-llvm ../../inputs/input_for_hello.c -o input_for_hello.ll
```



最后，用`libHelloWorld.dylib`作为插件，分析了`input_for_hello.ll`

```shell
llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ../lib/libHelloWorld.dylib -passes=hello-world -disable-output input_for_hello.ll
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
(llvm-tutor) Hello from: bar
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: fez
(llvm-tutor)   number of arguments: 3
(llvm-tutor) Hello from: main
(llvm-tutor)   number of arguments: 2
```

- `(llvm-tutor) Hello from: foo`

  `(llvm-tutor)   number of arguments: 1`

  这表示 Pass 找到了名为 `foo` 的函数，并检测到它有 1 个参数 (`int a`)。

- `(llvm-tutor) Hello from: bar`

  `(llvm-tutor)   number of arguments: 2`

  这表示 Pass 找到了名为 `bar` 的函数，并检测到它有 2 个参数 (`int a, int b`)。

- `(llvm-tutor) Hello from: fez`

  `(llvm-tutor)   number of arguments: 3`

  这表示 Pass 找到了名为 `fez` 的函数，并检测到它有 3 个参数 (`int a, int b, int c`)。

- `(llvm-tutor) Hello from: main`

  `(llvm-tutor)   number of arguments: 2`

  这表示 Pass 找到了名为 `main` 的函数，并检测到它有 2 个参数 (`int argc, char *argv[]`)。



## HelloWorld 源码