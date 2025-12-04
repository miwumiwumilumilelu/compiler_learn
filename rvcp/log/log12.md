# 2025.12.4 后端codegen——IR具体实现和CFG分析

今天实现Region成员函数，与BasicBlock的交互即链表的增删改查

这里的erase注意只需要对内部的基本块进行层层嵌套的erase即可

```c++
void Region::erase() {
    auto copy = bbs;
    for (auto bb : copy)
        bb->forceErase();
    parent->removeRegion(this);
    delete this;
}
```

新增op用**指定指令**插入的方式，但是基本块是固定形式的，创建的时候需要new

```c++
// step1: clear all preds and succs
// step2: for each block, look at its terminator to find its targets, and
//         add itself to the target's preds
// step3: for each block, for each of its preds, add the pred to its succs
void Region::updatePreds() {
    for (auto bb : bbs) {
        bb->preds.clear();
        bb->succs.clear();
    }
    for (auto bb : bbs) {
        assert(bb->getOpCount() > 0);
        auto last = bb->getLastOp();

        if (last->has<TargetAttr>()) {
            auto target = last->get<TargetAttr>();
            target->bb->preds.insert(bb);
        }

        if (last->has<ElseAttr>()) {
            auto ifnot = last->get<ElseAttr>();
            ifnot->bb->preds.insert(bb);
        }
    }

    for (auto bb : bbs) {
        for (auto pred : bb->preds) {
            pred->succs.insert(bb);
        }
    }
}
```

特别需要注意的是，更新区域内所有基本块的依赖时，需要把基本块的前驱后继先清空，然后根据terminal终止指令（只应该是goto/branch/return）来判断当前基本块的后继，此处用到类型判断，然后让后继的前驱连上当前基本块，最后对每一个基本块的前驱补充自己为其后继

总结就是，先连依赖设置pred，再通过preds设置succs

```c++
namespace {
// DFN is the number of each node in DFS order.
using DFN = std::unordered_map<BasicBlock*, int>;
using BBMap = std::unordered_map<BasicBlock*, BasicBlock*>;

using Vertex = std::vector<BasicBlock*>;

using SDom = BBMap;
using Parent = BBMap;
using UnionFind = BBMap;
using Best = BBMap;

int num = 0;
int pnum = 0;

// Dominators.
DFN dfn;
SDom sdom;
Vertex vertex;
Parent parents;
UnionFind uf;
Best best;
// Post-dominators. Just a copy-paste.
DFN pdfn;
SDom psdom;
Vertex pvertex;
Parent pparents;
UnionFind puf;
Best pbest;

void updateDFN(BasicBlock *current) {
    dfn[current] = num++;
    vertex.push_back(current);
    for (auto v : current->succs) {
        if (!dfn.count(v)) {
            parents[v] = current;
            updateDFN(v);
        }
    }
}

void updatePDFN(BasicBlock *current) {
    pdfn[current] = pnum++;
    pvertex.push_back(current);
    for (auto v : current->preds) {
        if (!pdfn.count(v)) {
        pparents[v] = current;
        updatePDFN(v);
        }
    }
}

BasicBlock* find(BasicBlock *v) {
    if (uf[v] != v) {
        BasicBlock *u = find(uf[v]);
        if (dfn[sdom[best[v]]] > dfn[sdom[best[uf[v]]]])
            best[v] = best[uf[v]];
        uf[v] = u;
    }
    return uf[v];
}

BasicBlock* pfind(BasicBlock *v) {
    if (puf[v] != v) {
        BasicBlock *u = pfind(puf[v]);
        if (pdfn[psdom[pbest[v]]] > pdfn[psdom[pbest[puf[v]]]])
            pbest[v] = pbest[puf[v]];
        puf[v] = u;
    }
    return puf[v];
}

void link(BasicBlock *v, BasicBlock *w) {
    uf[w] = v;
}

void plink(BasicBlock *v, BasicBlock *w) {
    puf[w] = v;
}

}
```

通过`https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/a%20fast%20algorithm%20for%20finding.pdf`

这篇论文，学习 **Lengauer-Tarjan 算法**来计算支配树和逆支配树

先定义一些基础的数据结构和算法组件——DFN标序，并查集压缩路径（正逆各自进行实现），合并支配树或逆支配树

其中：

`using UnionFind = BBMap;`——存储并查集结构中每个节点的**父节点**

`using Best = BBMap;`——对于节点 `v`，`Best[v]` 存储的是在并查集森林中，从 `v` 到其当前集合根节点的路径上，拥有最小DFN序且支配v的那个节点

```c++
// Use the Langauer-Tarjan approach.
// https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/a%20fast%20algorithm%20for%20finding.pdf
// Loop unrolling might update dominators very frequently, and it's quite time consuming.
void Region::updateDoms() {
}

void Region::updateDomFront() {
}

// A dual of updateDoms().
void Region::updatePDoms() {
}
```

