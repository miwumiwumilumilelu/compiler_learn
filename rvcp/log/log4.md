

# 2025.10.16 前端—— Parser 语法分析

## Parser.h

目的解决“必须在编译期确定数组维度”的 C 语言难题，使用常量折叠实现；并简化后续sema工作



在 Parser 产生完整的 AST 之前，常量折叠就开始了。这是为了解析数组的长度：

```c++
const int x[2] = { 1, 2 };
const int y[x[1]] = ...;
```

考虑到 `x[1]` 是编译期常量，这应当是合法的。为了正确产生 `y` 的类型，必须在这时就开始折叠，递归获取维度

```c++
// a compiler-time integer constant , used when early-folding array dimensions
class ConstValue {

    union {
        int *vi;
        float *vf;
    };

    std::vector<int> dims;

public:
    bool isFloat;
    ConstValue() {}
    ConstValue(int *vi, const std::vector<int> &dims): vi(vi), dims(dims), isFloat(false) {}
    ConstValue(float *vf, const std::vector<int> &dims): vf(vf), dims(dims), isFloat(true) {}

    ConstValue operator[](int i);
    int getInt();
    float getFloat();
    const std::vector<int>& getDims() const { return dims;}

    int size();
    int stride();

    // Copies a new memory. Doesn't release the original one.
    int *getRaw();
    float *getRawFloat();
    void *getRawRef() { return vi; }

    void release();

};
```

考虑情况：

1. ConstValue不仅可以是单个值，也可以是个数组，因此需要存储指针类型
2. ConstValue若是数组，则需要dims来存维度值
3. ConstValue需要支持数组访问，其中需要计算步长stride和数组大小size
4. 需要可以获取拷贝和指针，并可以手动管理内存



```c++
class Parser {
  using SymbolTable = std::map<std::string, ConstValue>;
    SymbolTable symbols;

    class SemanticScope {
      Parser &parser;
      SymbolTable symbols;
    public:
      SemanticScope(Parser &parser): parser(parser), symbols(parser.symbols) {};
      ~SemanticScope() { parser.symbols = symbols; }
    };
```

定义了一个仅用于常量的符号表（`map<string, ConstValue>`）

使用SymbolTable存储 `const` 变量或全局变量的值（作为 `ConstValue`），用于 `earlyFold`（常量折叠）

使用作用域管理类`SemanticScope`，针对`{}`，用于在解析 `block()` 时，自动管理 `symbols` 表的作用域



```c++
std::vector<Token> tokens;
size_t loc;
TypeContext &ctx;
std::string currentFunc;
```

定义解析器状态



```c++
Token last();
Token peek();
Token consume();

bool peek(Token::Type t);
Token expect(Token::Type t);

void printSurrounding();

template<class... Rest>
bool peek(Token::Type t, Rest... ts) {
    return peek(t) || peek(ts...);
}
template<class... T>
bool test(T... ts) {
    if (peek(ts...)) {
    loc++;
    return true;
    }
    return false;
}
```

Token辅助函数



```c++
// Parses only void, int and float.
Type *parseSimpleType();

// Const-fold the node.
ConstValue earlyFold(ASTNode *node);

ASTNode *primary();
ASTNode *unary();
ASTNode *mul();
ASTNode *add();
ASTNode *rel();
ASTNode *eq();
ASTNode *land();
ASTNode *lor();
ASTNode *expr();

ASTNode *stmt();
BlockNode *block();
TransparentBlockNode *varDecl(bool global);
FnDeclNode *fnDecl();
BlockNode *compUnit();

// Global array is guaranteed to be constexpr, so we return a list of int/floats.
// Local array isn't; so we return a list of ASTNodes.
void *getArrayInit(const std::vector<int> &dims, bool expectFloat, bool doFold);
```

声明语法规则

compUnit作为起始点，是递归下降中最顶层的规则

getArrayInit用来决定在编译期还是运行时进行常量折叠`earlyFold`

| **doFold**   | **doFold = true**   | **doFold = false** |
| ------------ | ------------------- | ------------------ |
| **何时使用** | 全局/`const` 数组   | 局部数组           |
| **调用函数** | `earlyFold(expr())` | `expr()`           |
| **返回类型** | `int*` 或 `float*`  | `ASTNode**`        |
| **存储节点** | `ConstArrayNode`    | `LocalArrayNode`   |
| **计算时间** | 编译期              | 运行时             |



```c++
public:
    Parser(const std::string &input, TypeContext &ctx);
    ASTNode *parse();
};
```

设置启动函数parse，`Parser.cpp` 中的实现将会调用 `compUnit()`，并最终返回整棵 AST 的根节点（一个 `BlockNode`）



## Parser.cpp（部分）

今天实现 `Parser` 的核心工具函数（常量折叠成员函数、Parser解析辅助函数、数组初始化函数）

```c++
#include "Parser.h"
#include "ASTNode.h"
// ... 

using namespace sys;

int ConstValue::size() { /* ... */ }
int ConstValue::stride() { /* ... */ }
std::ostream &operator<<(std::ostream &os, ConstValue value) { /* ... */ }
std::ostream &operator<<(std::ostream &os, const std::vector<int> vec) { /* ... */ }
int *ConstValue::getRaw() { /* ... */ }
float *ConstValue::getRawFloat() { /* ... */ }
void ConstValue::release() { /* ... */ }
int ConstValue::getInt() { /* ... */ }
float ConstValue::getFloat() { /* ... */ }
```

实现ConstValue 中所有的成员函数

`operator<<`：`ostream` 重载，纯粹为了调试时打印 `ConstValue` 和 `vector<int>`

其中

```c++
ConstValue ConstValue::operator[](int i) {
  assert(dims.size() >= 1);
  std::vector<int> newDims;
  newDims.reserve(dims.size() - 1);
  for (int j = 1; j < dims.size(); j++) {
      newDims.push_back(dims[j]);
  }
  return ConstValue(vi + i * stride(), newDims);
}
```

是常量折叠的核心，返回一个新的 `ConstValue` 对象，代表一个子数组（在原维度基础上去掉第一维，并计算新的数组头即vi指针）



```c++
Token Parser::last() { /* ... */ }
Token Parser::peek() { /* ... */ }
Token Parser::consume() { /* ... */ }
bool Parser::peek(Token::Type t) { /* ... */ }
Token Parser::expect(Token::Type t) { /* ... */ }
void Parser::printSurrounding() { /* ... */ }
```

`peek()`：“看”。返回 `tokens[loc]`。`loc` 不变。**test使用这个逻辑，检查符合进行loc++，否则return false**

`consume()`：“吃”。返回 `tokens[loc]` 并执行 `loc++`。

`expect(t)`：“期望”。它调用 `test(t)` 来消耗 Token。如果 `test` 失败，它会调用 `printSurrounding()`打印出错位置（`loc`）附近的 Token，并 `assert(false)`



```c++
Type *Parser::parseSimpleType() {
    if (test(Token::Int)) {
        return ctx.create<IntType>();
    } else if (test(Token::Float)) {
        return ctx.create<FloatType>();
    } else if (test(Token::Void)) {
        return ctx.create<VoidType>();
    } else {
        std::cerr << "expected type, but got " << peek().type << "\n";
        printSurrounding();
        assert(false);
    }
}
```

简单类型处理：使用 `ctx`来创建并返回相应的 `Type` 对象，前提是先检查Token



```c++
void *Parser::getArrayInit(const std::vector<int> &dims, bool expectFloat, bool doFold) {
  // ... (Lambda 辅助函数)
  
  // ... (内存分配)
  
  // ... (do-while 循环)
  
  return vi;
}
```

针对数组初始化，有多种情况：

1. 完全嵌套初始化，每一层数组都由 `{}` 包围
   - `int a[2][2] = { {1, 2}, {3, 4} };`

2. 花括号省略，初始化器会按顺序“拍平”并填充内存
   - `int a[2][2] = {1, 2, 3, 4};`

3. 不完整初始化，任何未显式初始化的元素都会被自动设为 0
   - `int a[3][3] = {1, 2};`

4. 不完整的嵌套子列表，当一个子列表 `}` 提前关闭时，`place` 游标必须“快进”到下一个子列表的开头
   - `int a[3][3] = { {1}, {2} };`

5. 规则 5：混合初始化，允许混合使用花括号和省略花括号
   - `int a[3][4] = {{1, 2, 3, 4}, 5, 6, {7}};`

接下来详细介绍各部分：

```c++
auto carry = [&](std::vector<int> &x) {
  for (int i = (int) x.size() - 1; i >= 1; i--) {
    if (x[i] >= dims[i]) {
      auto quot = x[i] / dims[i];
      x[i] %= dims[i];
      x[i - 1] += quot;
    }
  }
};

auto offset = [&](std::vector<int> &x) {
  int total = 0, stride = 1;
  for (int i = (int) x.size() - 1; i >= 0; i--) {
    total += x[i] * stride;
    stride *= dims[i];
  }
  return total;
};
```

**`[&]`**：**捕获列表** - 按引用捕获所有外部变量

**`(std::vector<int> &x)`**：参数列表

Offset将多维数组拍平成一维，计算出x的对应一维数组中的索引

carry处理进位，因为是最低一位++，所以可能需要进位

```c++
// initialize with 'dims.size()' zeroes.
std::vector<int> place(dims.size(), 0);
int size = 1;
for (auto x : dims)
  size *= x;

void *vi = !doFold
  ? (void*) new ASTNode*[size]
  : expectFloat ? (void*) new float[size] : new int[size];

memset(vi, 0, size * (doFold ? expectFloat ? sizeof(float) : sizeof(int) : sizeof(ASTNode*)));
```

处理局部数组和全局/常量数组两种情况，分配内存并置0

这里初始化了置0的place维度数组，后续会用它来解析循环

```c++
// add 1 to `place[addAt]` when we meet the next `}`.
int addAt = -1;
do {
  if (test(Token::LBrace)) {
    addAt++;
    continue;
  }

  if (test(Token::RBrace)) {
    if (--addAt == -1)
      break;

    // Bump `place[addAt]`, and set everything after it to 0.
    place[addAt]++;
    for (int i = addAt + 1; i < dims.size(); i++)
      place[i] = 0;
    if (!peek(Token::RBrace))
      carry(place);

    // If this `}` isn't at the end, then a `,` or `}` must follow.
    if (addAt != -1 && !peek(Token::RBrace))
      expect(Token::Comma);
    continue;
  }

  if (!doFold)
    ((ASTNode**) vi)[offset(place)] = expr();
  else if (expectFloat)
    ((float*) vi)[offset(place)] = earlyFold(expr()).getFloat();
  else
    ((int*) vi)[offset(place)] = earlyFold(expr()).getInt();

  place[place.size() - 1]++;

  // Automatically carry.
  // But don't carry if the next token is `}`. See official functional test 05.
  if (!peek(Token::RBrace))
    carry(place);
  if (!test(Token::Comma) && !peek(Token::RBrace))
    expect(Token::RBrace);
} while (addAt != -1);
```

这里针对之前提到各种情况，建立规则。如特别处理了初始化不完整的情况，则会进行

```c++
place[addAt]++;
for (int i = addAt + 1; i < dims.size(); i++)
  place[i] = 0;
```

实现未遍历到的直接跳过，仅跟随嵌套层级addAt，因为之前已经设置好了vi全部为0的初始化，所以放心，未显示出现的都默认为0了

在一个子列表`{...}`结束时，正确地“快进” `place` 索引游标，并为下一个元素的解析做好准备

场景示例：`int a[2][2][2] = { {{1}, {2}} };`

- `dims` = `{2, 2, 2}`
- Token 流: `{`, `{`, `{`, `1`, `}`, `,`, `{`, `2`, `}`, `}`, `}`

追踪：`Parser` 刚刚解析完 `1`。

- `place` (当前索引) 是: `{0, 0, 1}`
- `addAt` (嵌套深度) 是: `2` (在 `{{{` 内部)
- `loc` (当前 Token) 指向: `}`

`test(Token::RBrace)` 匹配成功。它消耗掉 `}`，`loc` 现在指向 `,`

`addAt` 从 `2` 变为 `1`。`1 == -1` 为 `false`，检查是否是**最外层**的 `}`，这里不是，所以继续

`place[addAt]++`，这里`place` 向量变为 `{0, 1, 1}`，紧接着将所有更深嵌套的索引重置为 `0`，`place` 向量变为 `{0, 1, 0}`

`place` 游标从 `[0, 0, 1]`（`a[0][0][1]`）快进到了 `[0, 1, 0]`（`a[0][1][0]`）。这跳过了 `a[0][0][1]`（它在 `memset` 时被设为 0），并为下一个 `{2}` 做好了准备

```c++
// test(Token::RBrace)的情况中会有以下判断
if (!peek(Token::RBrace))
  carry(place);

// If this `}` isn't at the end, then a `,` or `}` must follow.
if (addAt != -1 && !peek(Token::RBrace))
  expect(Token::Comma);
continue;
```

考虑存在如以下的情况:

> int a[3] [4] = {
>     {1, 2, 3, 4},      // 第 0 行填满（place = [0, 4]）// <- 这里遇到 } 并且下一个 token 是 , 
>     {5, 6},            // 第 1 行只填 2 个（place = [1, 2]） // <- 这里遇到 } 并且下一个 token 是 , 
>     }                 

即有个维度为空值出现 , } 结束的情况或者出现了填满的情况，此时进位

| 条件                   | 含义                                                         |
| ---------------------- | ------------------------------------------------------------ |
| [addAt != -1]          | 当前不是最外层（还有未闭合的 `{`），说明这个 `}` 不是整个初始化器的最后一个 `}` |
| `!peek(Token::RBrace)` | 下一个 token 不是 `}`（即下一个不是右花括号）, 不是 }} 的情况 |
| 两个条件都真           | 说明 `}` 后面还要继续初始化，必须有一个逗号                  |

```c++
if (!doFold)
    ((ASTNode**) vi)[offset(place)] = expr();
else if (expectFloat)
    ((float*) vi)[offset(place)] = earlyFold(expr()).getFloat();
else
    ((int*) vi)[offset(place)] = earlyFold(expr()).getInt();

place[place.size() - 1]++;
```

如果test既不是{ 也不是}，则说明是值（','在条件检查的过程中consume了`expect(Token::Comma)`）

| 条件                                                         | 操作                                                         | 用途                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| `!doFold`                                                    | 存为 [ASTNode*](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-browser/workbench/workbench.html) | **延迟求值**（保存抽象语法树节点，运行时再计算） |
| [doFold && expectFloat](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-browser/workbench/workbench.html) | 立即计算并存为 `float`                                       | **编译期常量折叠**（浮点常量）                   |
| 否则                                                         | 立即计算并存为 `int`                                         | **编译期常量折叠**（整数常量）                   |

类型转换：`((ASTNode**) vi)` / `((float*) vi)` / `((int*) vi)` —— 按类型强制转换通用指针

`place[place.size() - 1]++;`就是索引推进，因为有进位逻辑，所以按最低位进行++

考虑存在`int a[2][2] = { 1, 2, 3, 4 }`的情况，读值后需要自己判断并进行自动进位

```c++
// Automatically carry.
// But don't carry if the next token is `}`.
if (!peek(Token::RBrace))
    carry(place);
if (!test(Token::Comma) && !peek(Token::RBrace))
    expect(Token::RBrace);
```

如果当前token不是`,`且token也不是`}`，则期望`}`（如果test失败是不是loc++到下一个token的）

即在元素之间，必须出现 `,` 或 `}`，否则错误

