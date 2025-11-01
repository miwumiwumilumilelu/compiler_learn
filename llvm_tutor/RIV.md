## RIV

## RIV Pass使用——IR文件分析

**分析 Pass**，它的作用是找出在程序的每一个**基本块（Basic Block）**中，哪些整型变量是“可见”或“可达”的

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -emit-llvm -S -O1 ../inputs/input_for_riv.c -o input_for_riv.ll
```

opt操作，并调用配套的**打印 Pass** `print<riv>` 来将分析结果显示在终端上

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libRIV.dylib --passes="print<riv>" -disable-output input_for_riv.ll
=================================================
LLVM-TUTOR: RIV analysis results
=================================================
BB id      Reachable Integer Values      
-------------------------------------------------
BB %3                                         
             i32 %0                        
             i32 %1                        
             i32 %2                        
BB %6                                         
               %4 = add nsw i32 %0, 123    
               %5 = icmp sgt i32 %0, 0     
             i32 %0                        
             i32 %1                        
             i32 %2                        
BB %17                                        
               %4 = add nsw i32 %0, 123    
               %5 = icmp sgt i32 %0, 0     
             i32 %0                        
             i32 %1                        
             i32 %2                        
BB %10                                        
               %7 = mul nsw i32 %1, %0     
               %8 = sdiv i32 %1, %2        
               %9 = icmp eq i32 %7, %8     
               %4 = add nsw i32 %0, 123    
               %5 = icmp sgt i32 %0, 0     
             i32 %0                        
             i32 %1                        
             i32 %2                        
BB %14                                        
               %7 = mul nsw i32 %1, %0     
               %8 = sdiv i32 %1, %2        
               %9 = icmp eq i32 %7, %8     
               %4 = add nsw i32 %0, 123    
               %5 = icmp sgt i32 %0, 0     
             i32 %0                        
             i32 %1                        
             i32 %2               
```

随着程序控制流的推进，可达变量的集合是不断累积和扩大的



## RIV 源码

### .h

```c++
  using Result = llvm::MapVector<llvm::BasicBlock const *,
                                 llvm::SmallPtrSet<llvm::Value *, 8>>;
```

`SmallPtrSet`：8指针空间

基本块指针——>Value指针

```c++
struct RIV : public llvm::AnalysisInfoMixin<RIV> {
	...
	private:
  static llvm::AnalysisKey Key;
  ...
}
```

分析Pass对应的唯一标识，可以识别并缓存其分析结果

```c++
Result buildRIV(llvm::Function &F,
                  llvm::DomTreeNodeBase<llvm::BasicBlock> *CFGRoot);
```

核心函数，要用到CFGRoot



### .c

`using NodeTy = DomTreeNodeBase<llvm::BasicBlock> *;` 支配树节点指针

```c++
RIV::Result RIV::buildRIV(Function &F, NodeTy CFGRoot) {
  Result ResultMap;

  // Initialise a double-ended queue that will be used to traverse all BBs in F
  std::deque<NodeTy> BBsToProcess;
  BBsToProcess.push_back(CFGRoot);
  ...
}
```

从CFGroot开始，deque双端队列



**预计算每个基本块定义的整数值**

```c++
  // STEP 1: For every basic block BB compute the set of integer values defined
  // in BB
  DefValMapTy DefinedValuesMap;
  for (BasicBlock &BB : F) {
    auto &Values = DefinedValuesMap[&BB];
    for (Instruction &Inst : BB)
      if (Inst.getType()->isIntegerTy())
        Values.insert(&Inst);
  }
```

`auto &Values = DefinedValuesMap[&BB];`以`BB`的地址为键创建映射Map

`Values.insert()` 将是整型指令Inst的指针插入到当前基本块对应的 `SmallPtrSet` 集合中



**初始化入口的RIV集合**

```c++
  // STEP 2: Compute the RIVs for the entry BB. This will include global
  // variables and input arguments.
  auto &EntryBBValues = ResultMap[&F.getEntryBlock()];

  for (auto &Global : F.getParent()->globals())
    if (Global.getValueType()->isIntegerTy())
      EntryBBValues.insert(&Global);

  for (Argument &Arg : F.args())
    if (Arg.getType()->isIntegerTy())
      EntryBBValues.insert(&Arg);
```

对于入口基本块来说，添加全局变量和函数参数



**遍历支配树，传播RIV**

```c++
// 只要 BBsToProcess 列表不为空，就说明还有支配树节点没有被处理过，循环继续
while (!BBsToProcess.empty()) {
    auto *Parent = BBsToProcess.back();
    BBsToProcess.pop_back();
// 在处理子节点之前，我们先准备好所有需要“遗传”下去的数据
		// ParentDefs：获取 Parent 节点对应基本块定义的变量
		// ParentRIVs：获取 Parent 节点继承到之前的所有可见变量
    // Get the values defined in Parent
    auto &ParentDefs = DefinedValuesMap[Parent->getBlock()];
    // Get the RIV set of for Parent
    llvm::SmallPtrSet<llvm::Value *, 8> ParentRIVs =
        ResultMap[Parent->getBlock()];
// 这个for循环会遍历所有被 Parent 直接支配的 Child 节点
    // Loop over all BBs that Parent dominates and update their RIV sets
    for (NodeTy Child : *Parent) {
      BBsToProcess.push_back(Child);
      auto ChildBB = Child->getBlock();
      // Add values defined in Parent to the current child's set of RIV
      ResultMap[ChildBB].insert(ParentDefs.begin(), ParentDefs.end());
      // Add Parent's set of RIVs to the current child's RIV
      ResultMap[ChildBB].insert(ParentRIVs.begin(), ParentRIVs.end());
    }
  }
  return ResultMap;
}
```



**Pass 入口函数run**

```c++
RIV::Result RIV::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
  // 1. 请求依赖：向分析管理器请求 DominatorTree 的分析结果
  DominatorTree *DT = &FAM.getResult<DominatorTreeAnalysis>(F);
  // 2. 运行核心算法
  Result Res = buildRIV(F, DT->getRootNode());
  return Res;
}
```

