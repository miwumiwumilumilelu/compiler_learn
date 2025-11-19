# 2025.11.19 前端 —— fix bug & Sema

昨天遇到的bug，是因为Parser 中存在token 消耗顺序的不一致：

**`stmt()`中的 `test(Token::LBrace)`会消耗 `{` token!!!**

但 `block()`函数假设 `{` 还没被消耗，因此调用 `expect(Token::LBrace)`

这导致光标位置错误，期望的 token 和实际得到的 token 不匹配



两种修改方法

* 修改两个地方：

  1. **block() 函数**：

     ```c++
     BlockNode *Parser::block() {
       SemanticScope scope(*this);
     
       expect(Token::LBrace);
       std::vector<ASTNode *> nodes;
       
       while (!test(Token::RBrace))
         nodes.push_back(stmt());
     
       return new BlockNode(nodes);
     }
     ```

     * 移除 expect(Token::LBrace)

       ```c++
       BlockNode *Parser::block() {
           SemanticScope scope(*this);
           // LBrace has already been consumed by stmt()
           std::vector<ASTNode *> nodes;
         
           while (!test(Token::RBrace)) {
               nodes.push_back(stmt());
           }
         
           return new BlockNode(nodes);
       }
       ```

       添加注释说明 `{` 已由 stmt() 消耗

  2. **fnDecl() 函数**：

     - 添加` expect(Token::LBrace)`在调用 `block()`之前

       ```c++
           expect(Token::LBrace);  // Consume { before calling block()
           auto decl = new FnDeclNode(name, args, block());
           decl->type = ctx.create<FunctionType>(ret, params);
           return decl;
       }
       ```

       因为` fnDecl()`直接调用 `block()`，需要先消耗 `{`

  

* 修改stmt中的test为peek

  ```c++
  ASTNode *Parser::stmt() {
      if (test(Token::Semicolon)) {
          return new EmptyNode(); 
      }
  
      //debug: replace test with peek
      if (peek(Token::LBrace)) {
          return block();
      }
  ```

  为了保证block完整性，`{...}`我选择了方案2，之前的疏忽导致的错误

  

重新测试`basic.manbin`

```shell
compiler_learn/rvcp on  main [✘!?] via 🅒 base took 2.5s 
➜ ./src/build/test_parser ./test/custom/basic.manbin 
=== Parsing file: ./test/custom/basic.manbin ===

=== Parse Complete. AST Structure: ===

BlockNode (scoped)
  TransparentBlockNode (no scope)
    VarDeclNode (name: count, mut: 1, global: 1)
      (type: int)
      (init):
      IntNode (value: 0)
  FnDeclNode (name: main)
    (type: () -> int)
    (body):
    BlockNode (scoped)
      TransparentBlockNode (no scope)
        VarDeclNode (name: a, mut: 1, global: 0)
          (type: int)
          (init):
          IntNode (value: 7)
      WhileNode
        (cond):
        BinaryNode (kind: 8)
          VarRefNode (name: a)
          IntNode (value: 1)
        (body):
        BlockNode (scoped)
          AssignNode
            (left):
            VarRefNode (name: count)
            (right):
            BinaryNode (kind: 0)
              VarRefNode (name: count)
              IntNode (value: 1)
          IfNode
            (cond):
            BinaryNode (kind: 7)
              BinaryNode (kind: 4)
                VarRefNode (name: a)
                IntNode (value: 2)
              IntNode (value: 0)
            (ifso):
            BlockNode (scoped)
              AssignNode
                (left):
                VarRefNode (name: a)
                (right):
                BinaryNode (kind: 3)
                  VarRefNode (name: a)
                  IntNode (value: 2)
            (ifnot):
            BlockNode (scoped)
              AssignNode
                (left):
                VarRefNode (name: a)
                (right):
                BinaryNode (kind: 0)
                  BinaryNode (kind: 2)
                    VarRefNode (name: a)
                    IntNode (value: 3)
                  IntNode (value: 1)
      ReturnNode
        VarRefNode (name: count)

=== Cleaning up AST... ===
=== AST Cleaned Successfully. ===
```

所有测试通过！



## Sema.h

语义分析部分我只需用来确定并检查每个 AST 节点的类型及其相关约束

建立并维护符号表（当前作用域可见的变量/函数名 -> 类型），并处理作用域规则

最终目的是它为 CodeGen 提供已校验、带类型信息的 AST

```c++
namespace sys {

// We don't need to do type inference, hence no memory management needed
class Sema {
  TypeContext &ctx;
  // The current function we're in. Mainly used for deducing return type.
  Type *currentFunc;

  /*
  scope
  */
	/*
	func
	*/
public:
  // This modifies `node` inplace.
  Sema(ASTNode *node, TypeContext &ctx);
};

}
```

注意，Sema 不负责分配所有 Type 的所有权销毁（TypeContext 负责管理），其只是进行引用

`currentFunc`用于在遇到 `return` 语句时校验返回表达式类型是否匹配，做必要的转换；其较之前定义的，是Type * 类型，而不是std::string —— 这是因为 Type* 可以直接用于比较、转换检查和构造返回类型，避免查表开销`map<std::string , Type *>`

在构造 Sema 对象时，触发对整个 AST 的语义分析，进行AST节点的修改

**func:**

```c++
PointerType *decay(ArrayType *arrTy);
ArrayType *raise(PointerType *ptr);

Type *infer(ASTNode *node);
```

设计数组退化指针decay、指针进化数组raise、AST推导的成员函数

1. deacy: 保证当函数参数期待指针时，传入数组能够被正确视为指针；同时在类型推断中处理数组索引等情况
2. raise: 并非所有指针都能“安全”地提升为数组，raise 实现要有合理的前提检查或仅在特定模式下使用
3. infer: 这里特别简单设计了int2float 及 float2int 的二者类型转换规则

**scope:**

```c++
using SymbolTable = std::map<std::string, Type*>;
SymbolTable symbols;

class SemanticScope {
  Sema &sema;
  SymbolTable symbols;
public:
  SemanticScope(Sema &sema): sema(sema), symbols(sema.symbols) {}
  ~SemanticScope() { sema.symbols = symbols; }
};
```

- 构造时：复制当前 `sema.symbols`（map 的拷贝）
- 析构时：把 `sema.symbols` 恢复为原始拷贝

我们希望类似局部声明等有生命周期的变量或表达式不会冲突，不会污染外层符号表；退出时恢复外层表

这里与Parser中用于常量折叠的作用域一样借鉴了USTC_lab的作用域机制



