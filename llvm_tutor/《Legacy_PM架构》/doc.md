# Legacy PM架构



## 概述

中端和后端对Pass Manager有着多方面的需求，至少包括：

- 能将多个Pass按指定顺序组织成流水线，供实际优化流程运行调用。
- 能兼容应对中端和后端中的不同代码形态，这包括中端的IR形态和后端的Machine-IR形态。
- 能区分不同层级的代码单元，例如IR代码中的Module、CallGraphSCC、Function、BasicBlock几个代码层级。
- 能在接口定义上显式区分Analysis Pass和Transform Pass。
- 支持对Analysis Pass结果的获取和缓存，**支持在运行中将某个缓存的Analysis Pass标记为不再正确且需要重新运行**。
- 为中端、后端的Pass开发者定义**便捷的新Pass编写和注册标准化接口**，定义Analysis信息的获取接口，如ValueTracking。
- 性能方面。Pass Manager的整体设计，包括间依赖管理、Invalidate管理等等机制，在运行中不应造成可观的性能损失。



**Legacy PM框架实现**

Pass Manager是LLVM的核心机制，其中Legacy PM的框架代码都在llvm/lib/IR目录下，它们**在LLVM构建时被编译进libLLVMCore.a静态库**中。具体的，Legacy PM的实现代码包括以下.h和.cpp文件：

`llvm/include/llvm/IR/LegacyPassManager.h`

`llvm/include/llvm/IR/LegacyPassManagers.h`

`llvm/include/llvm/Pass.h`

`lib/IR/LegacyPassManager.cpp`



 LLVM 老架构里，**管理者自己本身也是个 Pass**。

比如

- **MPPassManager (Module级别管家)：** 它既派生自 `PMDataManager`（说明它是个管家，能管理一堆 Pass），又派生自 `Pass`。这意味着它在上一级眼里，就是一个普通的干活工人。
- **FPPassManager (Function级别管家)：** 同样地，它既管理着一堆 FunctionPass，但它自身却继承自 `ModulePass`。 

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/RkbPjPcGppicTjnhbTiaYyydd63XpGvG0TvlnYYpLFlkhjIfc8rcHfhMDlA2j1yDX9CiaS28FVCR6NBXR7jJFrjo49kibSylRHtlr9icxeibt5RpY/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

这几个xxxPass都直接派生自class Pass，**用于表示作用于不同IR单元的Pass**。

`ModulePass`的操作对象是整个IR Module，`CallGraphSCCPass`的操作对象是Module内若干个函数构成的**函数调用图强连通分量**，`FunctionPass`的操作对象是一个函数，不作跨函数优化，`LoopPass`的操作对象是函数内的一个循环语句，`RegionPass`操作对象是代码中一个单入口单出口的代码序列。这里所说的“操作”，可以是分析Analysis，也可以是Transform。

这几个类的入口函数名都形如runOnXXX()，具体的来说，分别是runOnModule()、runOnSCC()、runOnFunction()、runOnLoop()、runOnRegion()。从操作对象大小排序看，**ModulePass > CallGraphSCCPass > FunctionPass > LoopPass > RegionPass**

这些Pass的定义分布于头文件llvm/include/llvm/Pass.h和CallGraphSCCPass.h、LoopPass.h、RegionPass.h等当中



这里介绍一下CallGraphSCC：

**调用图 (Call Graph)：** 编译器在内存里画的一张图。图上的每一个“节点”是一个函数，每一条“有向边”代表一个函数调用了另一个函数（比如函数 A 内部写了 `B();`，就有一条从 A 指向 B 的边）。

**强连通分量 (SCC)：** 这是图论里的经典概念。在有向图中，如果有一堆节点，**它们彼此之间都可以顺着箭头互相到达**，这堆节点就被称为一个“强连通分量”。在 C 语言代码里，这就意味着**递归**或者**相互嵌套调用**。

- *简单递归：* 函数 A 自己调用自己，A 自身就是一个最小的 SCC。
- *交叉递归：* 函数 A 调用了 B，B 调用了 C，C 又掉头调用了 A。此时，A、B、C 三个函数死死地绑定在一起，它们构成了一个无法分割的 SCC。

LLVM 设计了 `CallGraphSCCPass`，它的操作对象不再是单一的函数，而是**把一整个 SCC（这几个互相调用的函数集群）当作一个整体单元**来处理 。如果 SCC 里只有一个独立的函数，它就处理这一个；如果 SCC 里有交叉递归的 3 个函数，它就将这 3 个函数放在一起进行联合分析和打破递归的优化。



## static成员地址来作PassID

在Legacy PM中，不同Analysis的区分是靠的依托这个const char*变量`PassID`实现的public方法`getPassID()`：

```c++
// llvm/include/llvm/Pass.h

class Pass {
  // . . .
  AnalysisID getPassID() const { return PassID; }
}
```

LLVM 放弃了用“字符串名字”或“枚举数字”来区分不同的 Pass

PassID的唯一性是**每个具体的Pass用一个static成员变量的地址**构造满足的，例如：

```c++
// llvm/lib/Transforms/Vectorize/LoadStoreVectorizer.cpp

class LoadStoreVectorizerLegacyPass : public FunctionPass {
public:
  static char ID; // 静态成员
  LoadStoreVectorizerLegacyPass() : FunctionPass(ID) { /* . . . */ }
  // . . .
};
char LoadStoreVectorizerLegacyPass::ID = 0;
```

FunctionPass()构造函数的入参ID会作为class Pass的构造函数入参，并赋给PassID。

在 C++ 中，`static` 静态变量存放在数据段（.data 或 .bss）。链接器（Linker）会向你做出绝对保证：**在整个程序的运行内存中，这个静态变量有且仅有一个绝对唯一的内存地址（比如 `0x7ffee104a8b0`）**

这里静态成员进行了《类内定义，类外初始化》：在 C++ 中，类的静态非 const 成员变量如果在头文件（.h）里直接初始化，那么当这个头文件被多个不同的 `.cpp` 文件包含（#include）时，每个 `.cpp` 文件在编译后都会生成一个该变量的实体。最终在把它们链接在一起时，链接器就会报“多重定义（Multiple Definition）”的错误。



## Pass的注册和执行

<img src="https://mmbiz.qpic.cn/mmbiz_png/RkbPjPcGpp9nicY7pvVV24KrqZicOmiawBINwibicX7NXIdBdRVX0zr7xNQRQyQgmGU67OBpEUAvDvteggLF8TdQstCl2v3iaf6yhiaIjtLlxQZrQY/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1" alt="图片" style="zoom:80%;" />

Pass管理的关键流程，一个是Pass的执行，另一个是Pass的注册。

Pass的执行流程，是指从Legacy PM框架的入口legacy::PassManager的`run()`入口如何一步步运行到某个特定的算法Pass；

Pass的注册流程，是指一个新Pass如何从legacy::PassManager的`add()`入口如何添加到这个Pass的直接管理者容器中的。这两个流程细化开来，和上面图中的数据结构间的关系是能对应上的。



我们现在就以 `LoadStoreVectorizerLegacyPass`（注意，它是一个 **FunctionPass**）为例：

**注册**

当在 LLVM 流水线中写下类似 `PM.add(new LoadStoreVectorizerLegacyPass())` 时，调度就开始了：

1. **`legacy::PassManager::add`** 

   接收这个新来的 Pass。但不负责具体分配，直接交给 `PassManagerImpl` 的 `add()`。

2. **`PMTopLevelManager::schedulePass`**

    拿到 `LoadStoreVectorizer` 后，不会立刻给它安排工位。

   - 它首先调用 `getAnalysisUsage()` 进行“政审”：你干活需要哪些前置情报？
   - 如果它发现向量化操作需要依赖“别名分析（Alias Analysis）”和“目标机器特征（TargetTransformInfo）”，就会**递归地**先去把这两个 Analysis Pass 注册进系统。这就保证了开工时情报绝对就绪。

3. **`FunctionPass::assignPassManager`**

    政审通过后，`LoadStoreVectorizer` 明确表示自己是一个 **FunctionPass**。大老板就会去当前的栈（`PMStack`）里找：现在有没有 `FPPassManager`（专门管函数级 Pass 的部门）？

   - 如果没有，大老板会当场 `new` 一个 `FPPassManager`，把它作为普通工人塞给更上层的 `MPPassManager`（模块级部门）。

4. **`PMDataManager::add`**

    找到了 `FPPassManager` 后，`LoadStoreVectorizer` 正式被加入到该领导维护的 `PassVector` 队列中。至此，注册完成！

**执行**

当所有 Pass 都注册完，顶层调用 `PM.run(Module)`

`PMTopLevelManager` 遍历手里所有的模块级管家（`PMDataManager`），挨个调用它们的 `runOnModule()`

`MPPassManager` 遍历自己手里的 Pass 队列。这时候，它点到了那个之前为了装 FunctionPass 而建立的“中层干部”——`FPPassManager`，并调用它的 `runOnModule()`

`FPPassManager` 拿到整个源文件（Module）后，做了一件极其关键的事：**它遍历了这个 Module 里的每一个 Function**（比如 `main` 函数、`foo` 函数）

对于当前拿到的具体函数（比如 `foo`），`FPPassManager` 开始遍历自己手里的 `FunctionPass` 队列



由于LLVM 15中后端仍然在用Legacy PM，可以用gdb抓取后端工具llc的调用栈。直接编译的命令行类似于：

> llc -mtriple=sparcv9 y8.ll

调用栈：

```c++
SparcDAGToDAGISel::runOnMachineFunction() at llvm/lib/Target/Sparc/SparcISelDAGToDAG.cpp
llvm::MachineFunctionPass::runOnFunction() at llvm/lib/CodeGen/MachineFunctionPass.cpp
llvm::FPPassManager::runOnFunction() at llvm/lib/IR/LegacyPassManager.cpp
llvm::FPPassManager::runOnModule() at llvm/lib/IR/LegacyPassManager.cpp
MPPassManager::runOnModule() at llvm/lib/IR/LegacyPassManager.cpp
llvm::legacy::PassManagerImpl::run() at llvm/lib/IR/LegacyPassManager.cpp
llvm::legacy::PassManager::run() at llvm/lib/IR/LegacyPassManager.cpp
compileModule() at llvm/tools/llc/llc.cpp
main() at llvm/tools/llc/llc.cpp
```

<img src="https://mmbiz.qpic.cn/sz_mmbiz_png/RkbPjPcGpp9uibndhUm8WkJvfe5p6rwBP6LnyCib8ZkXZrE34Mvs0v7bEa820ZsEEyMqqwcqupCccx9tXtf9LP5DoLP6llGKaibR7lDCSiakgQU/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2" alt="图片" style="zoom:67%;" />

- **循环A**：由PassManagerImpl通过PMTopLevelManager遍历它的PMDataManager*队列PassManagers，并挨个调用每个PMDataManager的runOnModule()。
- **循环B**：由MPPassManager通过PMDataManager遍历它的Pass*队列PassVector，并挨个将Pass强转为ModulePass然后调用它的runOnModule()。
- **循环C**：由FPPassManager（继承自ModulePass）遍历Module对象中的各个Function，并挨个调用runOnFunction()来访问这个Function。
- **循环D**：FPPassManager通过PMDataManager遍历它的Pass*队列PassVector，并挨个将Pass强转为FunctionPass然后调用它的runOnFunction()。

归纳起来，这几层循环所用到的一对多分层映射包括：

- PMTopLevelManager -> PMDataManager 映射：用了1次，循环A
- PMDataManager –> Pass 映射：用了2次，循环B、D
- Module -> Function 映射：用了1次，循环C