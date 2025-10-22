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



## InjectFuncCall 源码 

### .h

```c++
struct InjectFuncCall : public llvm::PassInfoMixin<InjectFuncCall> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};
```



### .cpp

```c++
bool InjectFuncCall::runOnModule(Module &M) {
  bool InsertedAtLeastOnePrintf = false; // Module 是否修改

  auto &CTX = M.getContext(); // 获取llvm全局API接口
  PointerType *PrintfArgTy = PointerType::getUnqual(CTX); // 定义char*，即llvm中的i8*

  // STEP 1: Inject the declaration of printf
  // ----------------------------------------
  FunctionType *PrintfTy = FunctionType::get(
      IntegerType::getInt32Ty(CTX),
      PrintfArgTy,
      /*IsVarArgs=*/true);

  FunctionCallee Printf = M.getOrInsertFunction("printf", PrintfTy);

  // Set attributes as per inferLibFuncAttributes in BuildLibCalls.cpp
  Function *PrintfF = dyn_cast<Function>(Printf.getCallee());
  PrintfF->setDoesNotThrow();
  PrintfF->addParamAttr(0, llvm::Attribute::getWithCaptureInfo(
                               M.getContext(), llvm::CaptureInfo::none()));
  PrintfF->addParamAttr(0, Attribute::ReadOnly);


  // STEP 2: Inject a global variable that will hold the printf format string
  // ------------------------------------------------------------------------
  llvm::Constant *PrintfFormatStr = llvm::ConstantDataArray::getString(
      CTX, "(llvm-tutor) Hello from: %s\n(llvm-tutor)   number of arguments: %d\n");

  Constant *PrintfFormatStrVar =
      M.getOrInsertGlobal("PrintfFormatStr", PrintfFormatStr->getType());
  dyn_cast<GlobalVariable>(PrintfFormatStrVar)->setInitializer(PrintfFormatStr);
```

定义 printf 的函数类型：` i32 printf(i8*, ...)`

给出了返回值类型i32，和入参char* 以及允许接收可变参数

```c++
  FunctionType *PrintfTy = FunctionType::get(
      IntegerType::getInt32Ty(CTX),
      PrintfArgTy,
      /*IsVarArgs=*/true);
```

在模块 M 中获取或插入 "printf" 函数，即**注入PrintfTy的声明**，用Printf接收

```c++
FunctionCallee Printf = M.getOrInsertFunction("printf", PrintfTy);
```

可选：为 printf 声明添加属性，帮助优化器

其中`dyn_cast`将Value *指针转成了其子类指针Function *    （getCallee()返回的是Value*）

```c++
Function *PrintfF = dyn_cast<Function>(Printf.getCallee());
PrintfF->setDoesNotThrow();
PrintfF->addParamAttr(0, llvm::Attribute::getWithCaptureInfo(
                               M.getContext(), llvm::CaptureInfo::none()));
PrintfF->addParamAttr(0, Attribute::ReadOnly);
```



创建LLVM常量`PrintfFormatStr`字符串，后在模块中创建或获取一个全局变量 "PrintfFormatStr" 来准备存储这个字符串

常量指针向下转型为全局变量GlobalVariable，并通过`setInitializer`设置初始值为之前创建的常量字符串

```cpp
  llvm::Constant *PrintfFormatStr = llvm::ConstantDataArray::getString(
      CTX, "(llvm-tutor) Hello from: %s\n(llvm-tutor)   number of arguments: %d\n");

  Constant *PrintfFormatStrVar =
      M.getOrInsertGlobal("PrintfFormatStr", PrintfFormatStr->getType());
  dyn_cast<GlobalVariable>(PrintfFormatStrVar)->setInitializer(PrintfFormatStr);
```

现在，IR 模块中有了一个全局的字符串常量，可以在函数中引用它



```c++
  // STEP 3: For each function in the module, inject a call to printf
  // ----------------------------------------------------------------
  for (auto &F : M) { // 遍历模块中的所有函数 F
    // 如果 F 只是一个声明 (没有函数体，如 extern void foo())，则跳过
    if (F.isDeclaration())
      continue;
    // 1. 获取一个 IRBuilder
    // 并将其 "插入点" 设置为函数 F 入口块的 "第一条指令之前"
    IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());
    // 2. 为当前函数名创建一个全局字符串 (作为 printf 的 %s 参数)
    auto FuncName = Builder.CreateGlobalString(F.getName());
    // 3. 获取格式化字符串的指针 (i8*)
    // 全局变量 PrintfFormatStrVar 的类型是 [N x i8] (数组)
    // printf 需要的是 i8* (指针)，所以需要一个 "cast" (类型转换)
    llvm::Value *FormatStrPtr =
        Builder.CreatePointerCast(PrintfFormatStrVar, PrintfArgTy, "formatStr");
    // 4. (调试信息)
    LLVM_DEBUG(dbgs() << " Injecting call to printf inside " << F.getName() << "\n");

    // 5. 创建对 printf 的调用指令！
    Builder.CreateCall(
        Printf, // 要调用的函数
        {FormatStrPtr,                  // 第一个参数: 格式化字符串 (i8*)
         FuncName,                      // 第二个参数: 函数名 (i8*)
         Builder.getInt32(F.arg_size()) // 第三个参数: 函数参数个数 (i32)
        });

    InsertedAtLeastOnePrintf = true; // 标记我们已经修改了代码
  } llvm::Value *FormatStrPtr =
        Builder.CreatePointerCast(PrintfFormatStrVar, PrintfArgTy, "formatStr");

  return InsertedAtLeastOnePrintf; // 返回是否修改了 Module
}
```



```cpp
PreservedAnalyses InjectFuncCall::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  // 如果代码被修改 (Changed=true)，则返回 none()，
  // 如果没修改，返回 all()，表示所有分析都还保留
  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
```



Pass末尾模板：

```c++
// 定义插件信息
llvm::PassPluginLibraryInfo getInjectFuncCallPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "inject-func-call", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            // 注册一个 "解析回调"
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  // 当命令行中出现 "inject-func-call" 时...
                  if (Name == "inject-func-call") {
                    // ...就向 Pass 队列中添加一个 InjectFuncCall 实例
                    MPM.addPass(InjectFuncCall()); 
                    return true;
                  }
                  return false;
                });
          }};
}

// 插件的入口点，LLVM 加载动态库时会查找这个函数
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getInjectFuncCallPluginInfo();
}
```

