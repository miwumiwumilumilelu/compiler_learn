# **2025.12.8 后端codegen——IR具体实现和CFG分析**

今日完成了活跃度分析 (Liveness Analysis)模块。该分析基于迭代数据流算法，能够精确计算每个基本块入口 (`LiveIn`) 和出口 (`LiveOut`) 处的活跃变量集合

此外，还完善了 `Region` 的可视化输出功能，支持打印控制流图（CFG）拓扑信息（前驱、支配边界、IDom）及活跃度信息的调试转储



**OpBase.cpp**

活跃变量基于数据流分析，用于求得变量生命周期，在优化中非常重要

```c++
// SSA Page 116.
// https://pfalcon.github.io/ssabook/latest/book-full.pdf
void Region::updateLiveness() {
// 1. 确保图连接关系是最新的
    updatePreds();

    // 2. 清空旧的计算结果，准备重新计算
    for (auto bb : bbs) {
        bb->liveIn.clear();
        bb->liveOut.clear();
    }

    // 定义三个辅助映射来存储局部信息
    std::map<BasicBlock*, std::set<Op*>> phis;          // 块内的 Phi 指令
    std::map<BasicBlock*, std::set<Op*>> upwardExposed; // 向上暴露的使用 (Use)
    std::map<BasicBlock*, std::set<Op*>> defined;       // 块内定义的变量 (Def/Kill)

    // 3. 遍历每个块，扫描指令，收集局部信息
    for (auto bb : bbs) {
        for (auto op : bb->getOps()) {
            // A. 单独处理 Phi 节点
            // Phi 节点比较特殊，它们的“定义”发生在块的最开头，
            // 而它们的“使用”发生在进入该块的前驱边的末端。
            if (isa<PhiOp>(op)) {
                phis[bb].insert(op);
                continue; // Phi 不参与普通的 Def/Use 计算
            }

            // B. 记录定义 (Def)
            // 这些变量在块内被赋值，会“杀死”(Kill) 从外部流入的同名活跃性
            defined[bb].insert(op);

            // C. 记录向上暴露的使用 (Upward Exposed Use)
            // 如果一个变量在被当前块定义之前就被使用了，
            // 那么它必须是从这个块的入口处“活着”进来的。
            for (auto value : op->getOperands()) {
                if (!defined[bb].count(value.defining))
                    upwardExposed[bb].insert(value.defining);
            }
        }
    }
```

这部分不涉及块与块之间的数据流动，只是基本块内的局部信息

这相当于计算**传统数据流分析**中的 $Gen$ 和 $Kill$ 集合

- `upwardExposed` 就是 $Gen$（产生了活跃性）
- `defined` 就是 $Kill$（阻断了活跃性）

```c++
std::deque<BasicBlock*> worklist;
    
    // 找到所有的出口块（没有后继的块，如 return 块）
    // 接下来的 do-while 循环进行全量遍历
    std::copy_if(bbs.begin(), bbs.end(), std::back_inserter(worklist), [&](BasicBlock *bb) {
        return bb->succs.size() == 0;
    });
```

因为是**后向数据流分析**（活跃性是从后往前传的），通常从出口块开始效率更高，在这里初始化求解器

```c++
bool changed;
    do {
        changed = false;
        for (auto bb : bbs) {
            auto liveInOld = bb->liveIn; // 记录旧值，用于判断是否收敛

            // === 计算 LiveOut ===
            std::set<Op*> liveOut;
            for (auto succ : bb->succs) {
                // 1. 基础传播：后继的 LiveIn 也是当前块的 LiveOut
                // 但是！必须剔除后继块中由 Phi 节点定义的变量。
                // 原因：Phi 定义发生在后继块的开头，它是那个块内部产生的新值，
                // 不是从当前块流过去的老值。
                std::set_difference(
                    succ->liveIn.begin(), succ->liveIn.end(),
                    phis[succ].begin(), phis[succ].end(),
                    std::inserter(liveOut, liveOut.end())
                );

                // 2. Phi 节点的特殊“使用” (PhiUses)
                // 如果后继有一个 Phi 节点： %val = phi [%a, bb_pre1], [%b, bb_pre2]
                // 并且当前块 bb 就是 bb_pre1。
                // 那么，变量 %a 在离开 bb 时必须是活跃的，因为它要喂给那个 Phi 节点。
                for (auto phi : phis[succ]) {
                    auto &ops = phi->getOperands();
                    auto &attrs = phi->getAttrs();
                    for (size_t i = 0; i < ops.size(); i++) {
                        // 检查 Phi 的这个操作数是不是来自当前块 bb
                        if (FROM(attrs[i]) == bb)
                            liveOut.insert(ops[i].defining);
                    }
                }
            }
            bb->liveOut = liveOut; // 更新当前块的 LiveOut
```

正常后向传播：

$$LiveOut[B] = \bigcup_{S \in successors(B)} LiveIn[S]$$

$$LiveIn[B] = USE[B] \cup (LiveOut[B] - DEF[B])$$

但是这里考虑了SSA形式的CFG图，因此需要更加细节考虑：

$LiveOut(B) = \bigcup_{S \in succ(B)} (LiveIn(S) - PhiDefs(S)) \cup PhiUses(B \to S)$

$$LiveIn(B) = (LiveOut(B) - Defs(B)) \cup UpwardExposed(B) \cup PhiDefs(B)$$

```c++
							for (auto phi : phis[succ]) {
                    auto &ops = phi->getOperands();
                    auto &attrs = phi->getAttrs();
                    for (size_t i = 0; i < ops.size(); i++) {
                        // 检查 Phi 的这个操作数是不是来自当前块 bb
                        if (FROM(attrs[i]) == bb)
                            liveOut.insert(ops[i].defining);
                    }
                }
```

此处还要注意使用操作数对应的`FROM(attrs[i])`来判断活跃变量来自哪个前驱基本块

```c++
						// === 计算 LiveIn ===
            bb->liveIn.clear();

            // 1. 穿透流量：(LiveOut - Defs)
            // 如果一个变量在出口处活着，且没在块内被重定义（Kill），
            // 那它一定是从入口处活着进来的（穿透了整个块）。
            std::set_difference(
                liveOut.begin(), liveOut.end(),
                defined[bb].begin(), defined[bb].end(),
                std::inserter(bb->liveIn, bb->liveIn.end())
            );

            // 2. 向上暴露的使用：UpwardExposed
            // 块内自己要用到的外部变量，必须活着进来。
            for (auto x : upwardExposed[bb])
                bb->liveIn.insert(x);

            // 3. Phi 定义：PhiDefs
            // 代码里把 Phi 定义也加入了 LiveIn。
            // 严格的 SSA 语义中 Phi 在边上，但实现上通常认为 Phi 在块头定义。
            // 将其加入 LiveIn 通常是为了寄存器分配时的冲突检测（Interference Graph）。
            for (auto x : phis[bb])
                bb->liveIn.insert(x);

            // === 收敛检查 ===
            // 如果计算出的 LiveIn 和上一轮不一样，说明信息还在传播
            // 需要继续循环 (changed = true)
            if (liveInOld != bb->liveIn)
                changed = true;
        }
    } while (changed); // 不动点算法 (Fixed-Point Algorithm)
}
```

根据当前块的 `LiveOut` 和局部属性计算$LiveIn$



为了验证以上活跃度分析的正确性，实现了一个专门的调试函数`showLiveIn()`



最后完善了 IR 的文本化输出功能，使其包含丰富的控制流信息，基于上次实现的`Op::dump`	

实现了支持递归调用 `op->dump`，展示嵌套区域结构的`Region::dump`



**今天涉及C++库函数：**

* `std::set_difference`——计算两个集合的差集

  如

  $LiveIn = LiveOut - Defs$

  ```c++
              std::set_difference(
                  liveOut.begin(), liveOut.end(),
                  defined[bb].begin(), defined[bb].end(),
                  std::inserter(bb->liveIn, bb->liveIn.end())
              );
  ```

* `std::copy_if`——带条件的复制

  `std::back_inserter`——后插到迭代器

  如

  将出口块（没有后继的块）加入队列

  ```c++
  std::copy_if(bbs.begin(), bbs.end(), std::back_inserter(worklist), [&](BasicBlock *bb) {
      return bb->succs.size() == 0;
  });
  ```

  但在今天的后续实现中发现其实用不到，因为我进行全量遍历了，后向分析

* `rbegin()` / `rend()`——反向迭代器

  在 Lengauer-Tarjan 算法中，**逆序遍历** DFS 节点（从大到小）

  