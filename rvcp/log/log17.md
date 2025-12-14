# 2025.12.10 后端Opt——Flatten CFG pass

后端寄存器分配和指令选择通常是在“基本块+ 跳转边”构成的图（CFG）上工作的，而不是嵌套的 Region

因此这里实现第一个pass——FlattenCFG，通过`--dump-cfg-ir`进行打印，类汇编风格

实现CFGIR，将IR展平，其中对IfOp、WhileOp进行处理，并进行多余跳转基本块进行优化，对直通基本块等进行关联

这里保证了包含terminal的基本块是goto/branch/return



## **pass.h & pass.cpp & LowerPasses.h** 

```c++
class Pass {
protected:
    ModuleOp *module; // The module to be optimized
    std::vector<FuncOp *> collectFuncs();
public:
    Pass(ModuleOp *module) : module(module) {}
    virtual ~Pass() {}

    //
    virtual std::string name() = 0;
    virtual std::map<std::string, int> stats() = 0;
    virtual void run() = 0;
};
```

```c++
std::vector<FuncOp*> Pass::collectFuncs() {
    std::vector<FuncOp*> result;

    auto region = module->getRegion();
    if (!region) return result;

    auto block = region->getFirstBlock();
    if (!block) return result;

    for (auto op : block->getOps()) {
        if (auto func = dyn_cast<FuncOp>(op)) {
            result.push_back(func);
        }
    }

    return result;
}
```

```c++
class FlattenCFG : public Pass {
public:
  FlattenCFG(ModuleOp *module): Pass(module) {}
  
  std::string name() override { return "flatten-cfg"; };
  std::map<std::string, int> stats() override { return {}; }; 
  void run() override;
};
```



## FlattenCFG.cpp

详见源码，在处理跳转冗余块时，使用了lambda表达式，对终结符指令的属性进行了引用



![log17_1](./img/log17_1.png)