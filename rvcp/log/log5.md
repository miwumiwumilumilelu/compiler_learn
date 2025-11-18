# 2025.11.17 前端——Add ASTNode Impl for Parser

## Parser.cpp（部分）

由定义的ASTNode.h为基础，写相关递归下降实现：

```ABAP
Parser 层级结构（从下到上）
─────────────────────────────────────
语句层（stmt）
    ↑ 处理：变量声明、控制流、赋值等
    │
表达式层（expr → lor）
    ↑ 处理：逻辑或、逻辑与、相等、关系、加减、乘除、一元、原子
    │
原子层（primary）
    ↑ 处理：字面量、标识符、括号、数组、函数访问
    │
词法层（Token）
    ↓ 来自 Lexer
```

```c++
ASTNode *Parser::primary() { /* ... */ }
```

从最小单元(优先级最高)开始，即最小无法拆解的表达式单元：primary

```
primary() 的职责：
├─ 字面量（LInt、LFloat）→ IntNode / FloatNode
├─ 标识符 → 三种情况：
│  ├─ 变量引用 → VarRefNode
│  ├─ 数组访问 → ArrayAccessNode（如果后跟 []）
│  └─ 函数调用 → CallNode（如果后跟 ()）
└─ 括号表达式 → expr()（递归）
```

编译器工作流从小到大：

> 42 + x * 3
> ↓ 先识别原子
> 42, x, 3（三个 primary）
> ↓ 再组装成表达式
> mul: x * 3
> add: 42 + (x * 3)



```c++
ASTNode *Parser::unary() { /* ... */ }
ASTNode *Parser::mul() { /* ... */ }
ASTNode *Parser::add() { /* ... */ }
ASTNode *Parser::rel() { /* ... */ }
ASTNode *Parser::eq() { /* ... */ }
ASTNode *Parser::land() { /* ... */ }
ASTNode *Parser::lor() { /* ... */ }
ASTNode *Parser::expr() { /* ... */ }
```

接下来构建表达式优先级链

```c++
优先级链（从弱到强）：
─────────────────────
expr()       level 1: 最弱（整个表达式）
 ↓
lor()        level 2: 逻辑或 (||)
 ↓
land()       level 3: 逻辑与 (&&)
 ↓
eq()         level 4: 相等性 (==, !=)
 ↓
rel()        level 5: 关系比较 (<, >, <=, >=)
 ↓
add()        level 6: 加减法 (+, -)
 ↓
mul()        level 7: 乘除法 (*, /, %)
 ↓
unary()      level 8: 一元 (-, +, !)
 ↓
primary()    level 9: 最强（原子）
```

> 例子 1：`2 + 3 * 4`
>
> 正确解析：2 + (3 * 4) = 2 + 12 = 14
>
> 步骤：
> 1. expr() → lor() → ... → add() 开始
> 2. add() 先调用 mul() 得到 2
> 3. 发现 + 号
> 4. 再调用 mul() 得到 (3 * 4) = 12
> 5. 结果：Add(2, 12)
>
> 错误做法（如果 add() 先解析 *）：
>   会得到 (2 + 3) * 4 = 20（错误！）

> 例子 2：`a || b && c`
>
> 正确解析：a || (b && c)
> 原因：& 的优先级高于 ||
>
> 为什么代码中 lor() 调用 land()？
> lor() {
>     auto n = land();        ← 先拿更高优先级的
>     while (test(Token::Or)) ← 再处理低优先级的 ||
> }
>
> 反过来就错了：
> wrong_lor() {
>     auto n = expr();         ← 无限递归或优先级混乱
> }

其中，注意实现**左结合**方式：

```c++
ASTNode *Parser::add() {
    auto n = mul();                    // 获取左操作数
    while (peek(Token::Plus, Token::Minus)) {
        switch (consume().type) {
        case Token::Plus:
            n = new BinaryNode(..., n, mul());  // 构建节点，左=n，右=新的mul()
            break;
        }
    }
    return n;
}
```

并且，对于比较运算符：

```c++
ASTNode *Parser::rel() {
    auto n = add();
    while (peek(Token::Le, Token::Lt, Token::Ge, Token::Gt)) {
        switch (consume().type) {
        case Token::Le:
            n = new BinaryNode(BinaryNode::Le, n, add());
            break;
        case Token::Lt:
            n = new BinaryNode(BinaryNode::Lt, n, add());
            break;
        case Token::Ge:
            // a >= b  -> canonicalize to (b <= a)
            n = new BinaryNode(BinaryNode::Le, add(), n);
            break;
        case Token::Gt:
            // a > b  -> canonicalize to (b < a)
            n = new BinaryNode(BinaryNode::Lt, add(), n);
            break;
        default:
            assert(false);
        }
    }
    return n;
}
```

为了减少MIR量，指令集复杂度，对`Ge`和`Gt`进行了规范化(用`Le`和`Lt`来表示)，在`ASTNode.h`中有提到

```c++
            // Take special care for _sysy_{start,stop}time.
            // Their line numbers are encoded in their names.
            std::string name = vs;
            if (name.rfind("_sysy_starttime_", 0) != std::string::npos) {
                name = "_sysy_starttime";
                args.push_back(new IntNode(strtol(vs + 16, NULL, 10)));
            }
            if (name.rfind("_sysy_stoptime_", 0) != std::string::npos) {
                name = "_sysy_stoptime";
                args.push_back(new IntNode(strtol(vs + 15, NULL, 10)));
            }
```
针对特殊的ident，用于调试

name.rfind("xxx", 0) name在起始位置0找到了xxx   
std::string::npos是 rfind 在“未找到”时返回的特殊值

_sysy_starttime_ 字符串有 16 个字符。vs + 16 会计算出一个新指针     
strtol(vs + 16, NULL, 10) str转int，以10进制解析


```c++
ASTNode *Parser::stmt() {
    // 优先级 1：简单终止符
    if (test(Token::Semicolon)) → EmptyNode
    
    // 优先级 2：代码块
    if (test(Token::LBrace)) → block()
    
    // 优先级 3-5：关键字语句
    if (test(Token::Return)) → ReturnNode
    if (test(Token::If)) → IfNode
    if (test(Token::While)) → WhileNode
    
    // 优先级 6：变量声明
    if (peek(Token::Const, Token::Int, Token::Float)) → varDecl()
    
    // 优先级 7：通用表达式/赋值语句
    auto n = expr();
    if (test(Token::Assign)) → AssignNode
    else → 表达式语句
}
```

最后是语句层，作为节点中的最高层，调用所有下层

```c++
stmt() 需要：
├─ expr()           处理表达式语句、赋值等
├─ varDecl()        处理变量声明
├─ block()          处理代码块
├─ 其他语句         if/while/return等
└─ 递归调用 stmt()  处理嵌套语句
```



```c++
BlockNode *Parser::block() { /* ... */ }
TransparentBlockNode *Parser::varDecl(bool global) { /* ... */ }
```

完成代码块编写，其中TransparentBlockNode需要考虑多种情况，包括是否初始化



设计：

```c++
int x = 1;
if (x > 0) {
    int y = x * 2 + 3;
    y = y - 1;
}
```

```
Parser::parse()
  ↓
stmt() 1️⃣
  ├─ peek(int) → true
  └─ varDecl(true)
     └─ 创建 VarDeclNode("x", IntNode(1), ...)

stmt() 2️⃣
  ├─ test(if) → true
  ├─ expr() → rel() → ... → primary() → x
  └─ IfNode(
       cond: BinaryNode(Gt, VarRefNode("x"), IntNode(0)),
       ifso: block() {
           stmt() 3️⃣
             └─ varDecl(false)
                └─ VarDeclNode("y", 
                     BinaryNode(Add,
                       BinaryNode(Mul, VarRefNode("x"), IntNode(2)),
                       IntNode(3)
                     ))
           
           stmt() 4️⃣
             └─ expr() + test(Assign)
                └─ AssignNode(
                     VarRefNode("y"),
                     BinaryNode(Sub, VarRefNode("y"), IntNode(1))
                   )
       }
     )
```



