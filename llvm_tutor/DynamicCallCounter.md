# DynamicCallCounter

## DynamicCallCounter Pass 使用 —— 运行时动态插桩

过程用于统计**运行时**（即执行过程中遇到的）函数调用次数。它通过插入每次函数**调用时都会执行的调用计数指令**来实现。仅统计在输入模块中定义的函数调用次数。 这次传递建立在 InjectFuncCall

```shell
~/projects/llvm-project/build/bin/clang -emit-llvm -c ../inputs/input_for_cc.c -o input_for_cc.bc
```

执行 `opt` 命令

```shell
~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libDynamicCallCounter.dylib --passes="dynamic-cc" input_for_cc.bc -o instrumented.bin
```

没有禁用输出，而是用 `-o` 标志将 `opt` **修改后**的 IR 保存到一个名为 `instrumented.bin` 的新文件中

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/lli instrumented.bin 
=================================================
LLVM-TUTOR: dynamic analysis results
=================================================
NAME                 #N DIRECT CALLS
-------------------------------------------------
bar                  2         
main                 1         
foo                  13        
fez                  1     
```

`StaticCallCounter` 报告 `foo` 被调用了 3 次，因为它在源代码中只看到了 3 个 `call @foo` 指令

`DynamicCallCounter` 报告 `foo` 被调用了 13 次。这是因为在 `input_for_cc.c` 的 `main` 函数中，有一个循环会额外调用 `foo` 10 次。这个运行时行为只有通过**动态插桩**才能捕捉到

```c++
llvm-tutor/build on  main [?] via 🅒 base 
➜ cat ../inputs/input_for_cc.c     
//=============================================================================
// FILE:
//      input_for_cc.c
//
// DESCRIPTION:
//      Sample input file for CallCounter analysis.
//
// License: MIT
//=============================================================================
void foo() { }
void bar() {foo(); }
void fez() {bar(); }

int main() {
  foo();
  bar();
  fez();

  int ii = 0;
  for (ii = 0; ii < 10; ii++)
    foo();

  return 0;
}
```



## DynamicCallCounter 源码

Transformation Pass

### .h

```c++
struct DynamicCallCounter : public llvm::PassInfoMixin<DynamicCallCounter> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};
```

以模块为处理粒度



### .c



**辅助函数：`CreateGlobalCounter`**

```c++
Constant *CreateGlobalCounter(Module &M, StringRef GlobalVarName) {
  auto &CTX = M.getContext();

  // This will insert a declaration into M
  Constant *NewGlobalVar =
      M.getOrInsertGlobal(GlobalVarName, IntegerType::getInt32Ty(CTX));

  // This will change the declaration into definition (and initialise to 0)
  GlobalVariable *NewGV = M.getNamedGlobal(GlobalVarName);
  NewGV->setLinkage(GlobalValue::CommonLinkage);
  NewGV->setAlignment(MaybeAlign(4));
  NewGV->setInitializer(llvm::ConstantInt::get(CTX, APInt(32, 0)));

  return NewGlobalVar;
}
```

声明：使用 `M.getOrInsertGlobal` 在模块中插入一个全局变量的声明

定义与初始化：**通过 `getNamedGlobal` 获取这个变量**，然后使用 `setInitializer` 等方法为其设置链接类型、对齐方式，并最重要地，将其初始值设为 0



1. **为每个函数植入“计数探针”**

   ```c++
   for (auto &F : M) {
       if (F.isDeclaration()) // 跳过只有声明的函数 (如 printf)
         continue;
   
       IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());
     
       // Create a global variable to count the calls to this function
       std::string CounterName = "CounterFor_" + std::string(F.getName());
       // 创建全局计数器
       Constant *Var = CreateGlobalCounter(M, CounterName);
       CallCounterMap[F.getName()] = Var;
   
     	// Create a global variable to hold the name of this function
       auto FuncName = Builder.CreateGlobalString(F.getName());
       FuncNameMap[F.getName()] = FuncName;
     
       // 注入 "counter++" 逻辑
       LoadInst *Load2 = Builder.CreateLoad(IntegerType::getInt32Ty(CTX), Var);
       Value *Inc2 = Builder.CreateAdd(Builder.getInt32(1), Load2);
       Builder.CreateStore(Inc2, Var);
   }
   ```

   遍历模块中的每一个已定义的函数，并在其入口处插入使其专属计数器递增的代码

   **创建全局计数器和函数名，进行Map映射存储：**

   **`IRBuilder<>`**：这是 LLVM 中用于生成 IR 指令的强大工具。通过 `Builder(&*F.getEntryBlock().getFirstInsertionPt())`，我们将“画笔”定位到了**函数的最开始`getEntryBlock().getFirstInsertionPt()`**，确保计数操作在函数体其他代码之前执行

   **实现计数器+1:**

   **`Builder.CreateLoad`, `CreateAdd`, `CreateStore`**：`IRBuilder` 提供了一系列像 `Create...` 这样的方法，让我们可以用一种接近 C++ 的方式来生成底层的 IR 指令。这三行代码共同实现了 `*counter = *counter + 1;` 的逻辑，即“计数器加一”。

   

2. **准备打印工具 (`printf` 和格式化字符串)**

   ```c++
   PointerType *PrintfArgTy = PointerType::getUnqual(Type::getInt8Ty(CTX));
   FunctionType *PrintfTy =
       FunctionType::get(IntegerType::getInt32Ty(CTX), PrintfArgTy,
                           /*IsVarArgs=*/true);
   // 步骤 2: 注入 printf 函数的声明
   FunctionCallee Printf = M.getOrInsertFunction("printf", PrintfTy);
   // Set attributes as per inferLibFuncAttributes in BuildLibCalls.cpp
   Function *PrintfF = dyn_cast<Function>(Printf.getCallee());
   PrintfF->setDoesNotThrow();
   PrintfF->addParamAttr(0, llvm::Attribute::getWithCaptureInfo(
                                  M.getContext(), llvm::CaptureInfo::none()));
   PrintfF->addParamAttr(0, Attribute::ReadOnly);
   ```

   ```c++
   // 步骤 3: 注入将要用到的格式化字符串
   llvm::Constant *ResultFormatStr =
         llvm::ConstantDataArray::getString(CTX, "%-20s %-10lu\n");
   Constant *ResultFormatStrVar =
         M.getOrInsertGlobal("ResultFormatStrIR", ResultFormatStr->getType());
   std::string out = "";
   out += "=================================================\n";
   out += "LLVM-TUTOR: dynamic analysis results\n";
   out += "=================================================\n";
   out += "NAME                 #N DIRECT CALLS\n";
   out += "-------------------------------------------------\n";
   
   llvm::Constant *ResultHeaderStr =
       llvm::ConstantDataArray::getString(CTX, out.c_str());
   
   Constant *ResultHeaderStrVar =
       M.getOrInsertGlobal("ResultHeaderStrIR", ResultHeaderStr->getType());
   dyn_cast<GlobalVariable>(ResultHeaderStrVar)->setInitializer(ResultHeaderStr);
   ```

   

3. *Define a printf wrapper that will print the results*

   ```c++
     std::string out = "";
     out += "=================================================\n";
     out += "LLVM-TUTOR: dynamic analysis results\n";
     out += "=================================================\n";
     out += "NAME                 #N DIRECT CALLS\n";
     out += "-------------------------------------------------\n";
   
     llvm::Constant *ResultHeaderStr =
         llvm::ConstantDataArray::getString(CTX, out.c_str());
   
     Constant *ResultHeaderStrVar =
         M.getOrInsertGlobal("ResultHeaderStrIR", ResultHeaderStr->getType());
     dyn_cast<GlobalVariable>(ResultHeaderStrVar)->setInitializer(ResultHeaderStr);
   
   ```

   表头对应的输出

   out.c_str()

   setInitializer(ResultHeaderStr)

   ```c++
   // 在模块中定义一个新的、名为 "printf_wrapper" 的函数
   Function *PrintfWrapperF = ... M.getOrInsertFunction("printf_wrapper", ...);
   
   // 为这个新函数创建一个入口基本块并用 IRBuilder 填充内容
   llvm::BasicBlock *RetBlock = BasicBlock::Create(CTX, "enter", PrintfWrapperF);
   IRBuilder<> Builder(RetBlock);
   
   // 将“字符串数组”转换成 printf 函数能够理解的“字符串指针”
   llvm::Value *ResultHeaderStrPtr =
       Builder.CreatePointerCast(ResultHeaderStrVar, PrintfArgTy);
   llvm::Value *ResultFormatStrPtr =
       Builder.CreatePointerCast(ResultFormatStrVar, PrintfArgTy);
   
   // 生成一系列 printf 调用
   Builder.CreateCall(Printf, {ResultHeaderStrPtr}); // 打印表头
   for (auto &item : CallCounterMap) {
       LoadCounter = Builder.CreateLoad(...); // 从全局计数器加载值
       Builder.CreateCall(Printf, {..., LoadCounter}); // 打印一行统计
   }
   
   Builder.CreateRetVoid(); // 添加返回指令
   ```



**Pass 注册:**

```c++
llvm::PassPluginLibraryInfo getDynamicCallCounterPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "dynamic-cc", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "dynamic-cc") {
                    MPM.addPass(DynamicCallCounter());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getDynamicCallCounterPluginInfo();
}
```

**返回 `PreservedAnalyses::none()`**: 这是**关键**。因为它修改了 IR（插入了全局变量、指令和函数），所以它必须返回 `PreservedAnalyses::none()`，告诉 Pass 管理器：“我修改了代码，所有之前的分析结果都可能失效了！

 最后的 `getDynamicCallCounterPluginInfo` 和 `llvmGetPassPluginInfo` 是标准 boilerplate 代码，负责将这个 Pass 注册到 LLVM，并使其可以通过命令行参数 `--passes="dynamic-cc"` 来调用
