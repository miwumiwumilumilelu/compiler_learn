# 2025.12.3 后端codegen——IR具体实现和CFG分析

attr使用clone，将所有权转移，进行独立副本的创建，让每个Op都有自己的attr副本，避免共享问题，避免多个内存空间共享同一个指针，导致生命周期错乱。其中使用refcnt进行计数，通过Def-Use来判断变量活跃区间

Use-Def 链的一致性，在 `replaceOperand` 和 `setOperand` 中，必须非常小心地先移除旧的 Use 关系，再添加新的 Use 关系，否则会导致链表损坏

```c++
std::map<BasicBlock*, int> sys::bbmap;
int sys::bbid = 0;

static int getBlockID(BasicBlock *bb) {
  if (!bbmap.count(bb))
    bbmap[bb] = bbid++;
  return bbmap[bb];
}
```

对bb进行计数，映射区分

```c++
void BasicBlock::insert(iterator at, Op *op) {
  op->parent = this;
  op->place = ops.insert(at, op);
}

void BasicBlock::insertAfter(iterator at, Op *op) {
  op->parent = this;
  // insert before std::list::end() iterator
  if (at == ops.end()) {
    ops.push_back(op);
    op->place = --end();
    return;
  }
  op->place = ops.insert(++at, op);
}

void BasicBlock::remove(iterator at) {
  ops.erase(at);
}

BasicBlock* BasicBlock::nextBlock() const {
    auto it = place;
    return *++it;
}
```

对Basic和Op互联的基础操作实现（即基本块与指令链表管理）



然后是按照Op成员函数顺序，实现Use-Def链的管理和维护、内存管理、IR结构修改

以及下面的Op dump调试

```c++
static std::map<Op*, int> valueName = {};
static int id = 0;
```

为指令进行编号，初始id为0

```c++
std::string getValueNumber(Value value) {
    if (!valueName.count(value.defining))  // 如果Op还没有编号
        valueName[value.defining] = id++;   // 分配一个新编号
    return "%" + std::to_string(valueName[value.defining]);  // 返回"%N"格式
}
```

Op dump:

```c++
void Op::dump(std::ostream &os, int depth) {
    indent(os, depth * 2);  // 根据深度缩进
    os << getValueNumber(getResult()) << " = " << opname;  // 结果编号 = 操作名
    if (resultTy == Value::f32)  // 如果是浮点类型
        os << ".f";  // 添加类型后缀
    for (auto &operand : operands)  // 打印所有操作数
        os << " " << getValueNumber(operand);
    for (auto attr : attrs)  // 打印所有属性
        os << " " << attr->toString();
    if (regions.size() > 0) {  // 如果有子区域
        os << " ";
        for (auto &region : regions)
            region->dump(os, depth + 1);  // 递归打印子区域
    }
    os << "\n";  // 换行
}
```

包含depth嵌套，操作数和属性的情况（attr->toString()中已规范用<>）：

```
%3 = vector_loop.f %1 %2 <100> <name = test> {
      bb0:
        %4 = load.f %1 <size = 4>
        %5 = addf.f %4 %2
        %6 = yield %5
    }
```

