![image-20230913164805071](https://ustc-compiler-principles.github.io/2023/lab1/assets/image-20230913164805071-1694594895535-1-1694594931543-3.png)

可以看出核心程序是Bison源文件即parser.y——在其中定义main函数

## lex.l

**声明部分**包含名称声明和选项设置，`%{` 和 `%}` 之间的内容会被原样复制到生成的 C 文件头部，可用于编写 C 代码，如头文件声明和变量定义等

```
%option yylineno
%{
#include "parser.tab.h"
#include "string.h"
#include "def.h"

void yyerror(const char* fmt, ...);

int yycolumn = 1;

#define YY_USER_ACTION \
    yylloc.first_line = yylloc.last_line = yylineno; \
    yylloc.first_column = yycolumn; \
    yylloc.last_column = yycolumn + yyleng - 1; \
    yycolumn += yyleng;
%}
```

`%option yylineno`

 Flex 的选项开关，维护一个全局变量 `yylineno`，让它始终等于当前正在读取的行号



`parser.tab.h` 是由 Bison 自动生成的

`yylloc` 是由 Bison (在`parser.y` 文件中) 自动声明和定义的

```
// parser.y
%locations
```

当 Bison 看到了 `%locations` 这个指令后：

在 `parser.tab.h` (头文件) 中

- 会定义一个名为 `YYLTYPE` 的结构体，这个结构体通常长这个样子

- 它会**声明**一个 `extern` 的全局变量，这个变量的类型就是 `YYLTYPE`，名字就是 `yylloc`：

  `extern YYLTYPE yylloc;`



**规则部分**位于两个 `%%` 分隔符之间，词法分析器 (Lexer) 会逐一尝试将输入文本与这些模式进行匹配，一旦匹配成功，就执行对应的 C 代码动作

```
"="       { yylval.node = createNode(NODE_ASSIGN); return ASSIGNOP; }
"+"       { yylval.node = createNode(NODE_PLUS); return PLUS; }
... (所有其他运算符, 如 *, ==, >) ...
"<="      { yylval.node = createNode(NODE_LE); return LE; }
```

`createNode(NODE_ASSIGN)`：创建一个代表“赋值”的 AST 节点

`yylval.node = ...`：将这个新创建的节点放入 `yylval` 联合体中，传递给语法分析器

`return ASSIGNOP;`：告诉语法分析器：“我找到了一个赋值运算符，并且我把它的 AST 节点也顺便做好了，放在 `yylval` 里了。”

```
{int}     { yylval.type_int = atoi(yytext); return INT; }
{float}   { yylval.type_float = atof(yytext); return FLOAT; }
{id}      { strncpy(yylval.type_id, yytext, 31); ... return ID; }
```

这组规则用于匹配那些**自身带有值**的 Token

`yytext` 是一个 Flex 提供的特殊变量，它是一个 `char*` 指针，指向刚刚匹配到的字符串（例如 `"123"` 或 `"myVar"`）

动作代码（如 `atoi(yytext)`）将这个字符串转换成它实际的值（如整数 `123`）

`yylval.type_int = ...`：将这个转换后的值存入 `yylval` 联合体，传递给语法分析器

`return INT;`：告诉语法分析器：“我找到了一个整数，它的值在 `yylval.type_int` 里。”

```
[ \t]     { /* 忽略空格和制表符 */ }
\n        { yycolumn = 1; /* 遇到换行符，重置列号 */ }
```

```
.         { yyerror("Invalid character '%s'", yytext); }
```





## parser.y

**声明部分**

```
%{
// ... (includes) ...
extern int yylineno;
extern char *yytext;
extern FILE *yyin;
void yyerror(const char* fmt, ...);
int yylex();

struct ASTNode *root = NULL;
// ... (所有 createNode, addChild 函数) ...
%}
```

这部分代码会被原样复制到生成的 `parser.tab.c` 文件的顶部

**`extern` 声明**: 这是与 `lex.l` 的关键链接

- `yylineno` 和 `yytext`: 告诉Bison，`lex.l` 会提供这两个全局变量（当前行号和当前匹配的文本）
- `yyin`: 告诉Bison，我们会（在 `main` 中）设置这个文件指针，`lex.l` 将从这里读取输入。
- `int yylex();`: 这是**核心承诺**。`parser.y` 在此声明：“我保证会有一个名为 `yylex` 的函数（由 `lex.l` 提供），我将调用它来获取下一个Token。”

**`strdup`**

`value` 很可能指向 `yytext`（`lex.l` 中的一个缓冲区）。`lex.l` 在下次匹配到新 Token 时会**覆盖** `yytext` 的内容。`strdup` 会**复制**一份字符串的副本并存起来，这样即使 `yytext` 后来变了，我们保存的节点值也不会被破坏

这里采用儿子，兄弟表示

```c++
// 添加子结点
void addChild(struct ASTNode *parent, struct ASTNode *child) {
    if (parent->child == NULL) {
        parent->child = child;
    } else {
        struct ASTNode *temp = parent->child;
        while (temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = child;
    }
}
```



**语法规则部分**

`%union` 定义了语法符号的语义值类型的集合。

在 Bison 中，每个符号，包括记号和非终结符，都有一个不同数据类型的语义值，并且这些值通过 yylval 变量在移进和归约操作中传递。默认情况下，YYSTYPE（宏定义）为 yylval 的类型，通常为 int。但通过使用 `%union`，你可以重新定义符号的类型。使用 union 是因为不同节点可能需要不同类型的语义值

```c++
%union {
    int type_int;
    float type_float;
    char type_id[32];
    struct ASTNode *node;
}
```

`%token` - 定义“词汇表”（终结符）, 声明所有合法的“**词法单元**”（Tokens）

根据任务要求：——

注：这里约定通过bison中的%token为文法终结符定义的符号常量：
LC   {    RC   }    LP  （    RP   ）    SEMI  ；
INT  整型常量    FLOAT  浮点常量    ID  标识符
IF  if    ELSE  else    WHILE  while
ASSIGNOP  =   PLUS  +   MINUS  -   STAR  *   DIV  /
EQ  ==   NE  !=   LT  <   LE  <=   GT  >   GE  >

- `%token` 定义终结符。定义形式：`%token TOKEN1 TOKEN2`。一行可定义多个终结符，空格分隔。一般约定终结符都是大写，非终结符的名字是小写

- `%type` 定义非终结符

```
%token <type_int> INT
%token <type_float> FLOAT
%token <type_id> ID
%token IF ELSE WHILE SCAN PRINT
%token LC RC LP RP SEMI
%token <node> ASSIGNOP PLUS MINUS STAR DIV
%token <node> EQ NE GT GE LT LE
```

设计五个半成品也包装为node节点

`%type <node> program StmList Stmt CompSt Exp` 

```
// 定义运算符优先级
%nonassoc LOWER_THAN_ELSE
%nonassoc ELSE

%left ASSIGNOP
%left EQ NE GT GE LT LE
%left PLUS MINUS
%left STAR DIV
%right UMINUS
```

解决语法中的歧义

`%left`、`%right`、`%nonassoc` 定义终结符的结合性和优先级关系。定义形式与 `%token` 类似。先定义的优先级低，最后定义的优先级最高，同时定义的优先级相同。`%left` 表示左结 合（如“+”、“-”、“*”、“/”）；`%right` 表示右结合（例如“=”）；`%nonassoc` 表示不可结合（即它定义的终结符不能连续出现。例如“-”负号。如下定义中，优先级关系为：AA = BB < CC < DD；表示结合性为：AA、BB 左结合，DD 右结合，CC 不可结合

```
%left AA BB
%nonassoc CC
%right DD
```

`%start program`

作为起始规则，在分析Token时，唯一目标就是设法将整个输入文件最终**归约**成一个 `program` 符号

根据任务给出的语言文法：

```c++
program::=StmList
StmList::=空 | Stmt StmList
CompSt::=LC StmList RC
Stmt::=Exp SEMI          //表达式语句
    | CompSt           //复合语句
    | SCAN ID SEMI       //输入语句
    | PRINT Exp SEMI      //输出语句
    | IF LP Exp RP Stmt    //条件语句
    | IF LP Exp RP Stmt ELSE Stmt
    | WHILE LP Exp RP Stmt    //循环语句
Exp::=INT | FLOAT| ID    //整型常量，浮点常量、变量
    | Exp ASSIGNOP Exp     //赋值表达式
    | Exp PLUS Exp        //+-*/四则运算
    | Exp MINUS Exp 
    | Exp STAR Exp
    | Exp DIV Exp
    | Exp EQ Exp        //关系运算符
    | Exp NE Exp 
    | Exp GT Exp 
    | Exp GE Exp 
    | Exp  LT Exp
    | Exp LE  Exp
    | MINUS Exp        //单目-运算
    | LP Exp RP        
```

得出规则部分

举例：

`Stmt::=Exp SEMI`

```
	$$ (返回值)
        |
   NODE_EXP_STMT
        |
      child
        |
       $1 (来自 Exp 规则的 $$)
        |
       ...
```

```
Stmt: Exp SEMI { 
        $$ = createNode(NODE_EXP_STMT);
        addChild($$, $1);
      }
```



**程序入口部分**

**Main函数和错误处理函数**

`yyparse();`

yyparse() 是由 bison 自动生成的“语法分析总控函数”

它会开始工作，并在需要 Token 时自动调用 lex.l 里的 yylex() 

这个函数会一直运行，直到整个文件被成功解析或遇到语法错误    

```
// 引入 <stdarg.h> 头文件，这是使用 C 语言“可变参数”功能所必需的
#include <stdarg.h> 

/**
 * @brief yyerror - Bison/Yacc 约定的错误报告函数。
 * 当 yyparse() 遇到语法错误时，它会自动调用此函数。
 *
 * @param fmt Bison 传递过来的格式化错误字符串 (如 "syntax error, unexpected %s")
 * @param ... (省略号) C 语言的可变参数语法，用于接收 fmt 字符串中 %s, %d 等所需的额外参数
 */
void yyerror(const char* fmt, ...)
{
    // 1. 定义一个 va_list 类型的变量 'ap' (argument pointer)
    //    你可以把它想象成一个“篮子”，用来存放所有通过 "..." 传入的参数
    va_list ap;

    // 2. 初始化“篮子” ap。
    //    va_start 宏需要两个参数：
    //    - ap: 要初始化的篮子
    //    - fmt: ... 省略号之前的最后一个已知参数的名称
    //    这行代码让 ap 指向第一个可变参数
    va_start(ap, fmt);

    // 3. 打印自定义的错误前缀到“标准错误流”(stderr)
    //    - stderr 是专门用于输出错误信息的通道
    //    - yylloc 是由 %locations 启用、由 lex.l 填充的全局变量
    //    - 这行代码实现了“在xx行,xx列发生错误”的精确定位
    fprintf(stderr, "Grammar Error at Line %d Column %d: ", yylloc.first_line, yylloc.first_column);
    
    // 4. 打印 Bison 传递过来的核心错误信息
    //    - vfprintf 是 fprintf 的“可变参数列表”版本
    //    - 它会根据 fmt 字符串，从 ap“篮子”中取出所有参数，并格式化输出到 stderr
    vfprintf(stderr, fmt, ap);

    // 5. 打印导致错误的那个 Token 的文本
    //    - yytext 是由 lex.l 提供的全局变量，指向当前 Token 的字符串
    //    - 这能让用户直观地看到是哪个“词”引发了错误
    fprintf(stderr, " near '%s'\n", yytext);
    
    // 6. 清理“篮子”
    //    - va_end 必须与 va_start 成对出现，是 C 语言可变参数的收尾工作
    va_end(ap);
}
```



## displayAST.c

用凹入表的方式输出语法树的遍历，缩进空格数为4，浮点数显示按默认小数点后6位

```c
// 打印缩进的辅助函数
void print_indent(int indent) {
    for (int i = 0; i < indent; i++) {
        printf("    ");
    }
}
```

`if (node == NULL) return;`一次处理一个节点，结束递归

```c
    // 语句列表特殊处理：直接遍历打印其下的各个语句
    if (node->nodeType == NODE_STMT_LIST) {
        struct ASTNode *child = node->child;
        while (child != NULL) {
            displayAST(child, indent);
            child = child->next;
        }
        return;
    }
```

如果当前节点是 `NODE_STMT_LIST`，它会跳过打印自己

它会立即遍历它的所有子节点（即 `Stmt` 语句节点），并以相同的缩进级别 (`indent`) 递归调用 `displayAST`

记得return

switch - case 打印逻辑

1. 叶子结点：

   ```c
   				// 常量和变量
           case NODE_INT:
               printf("整型常量：%d\n", node->intValue);
               return;
           case NODE_FLOAT:
               printf("浮点常量：%f\n", node->floatValue);
               return;
           case NODE_ID:
               printf("变量：%s\n", node->stringValue);
               return;
           case NODE_SCAN_STMT:
               printf("输入变量：%s\n", node->child->stringValue);
               return; // 已处理子节点，直接返回
   ```

2. 复杂分支节点:

   由预期输出得知：

   ```
   预期输出：
   表达式语句：
       =
           变量：a
           浮点常量：10.100000
   输入变量：b
   条件语句(if_then_else)：
       条件：
           >
               变量：a
               变量：b
       if子句：
           表达式语句：
               =
                   变量：max
                   变量：a
       else子句：
           表达式语句：
               =
                   变量：max
                   变量：b
   输出表达式:
       变量：max
   
   
   ```

   ```c
   case NODE_IF_ELSE_STMT:
               printf("条件语句(if_then_else)：\n");
               print_indent(indent + 1);
               printf("条件：\n");
               displayAST(node->child, indent + 2); // 手动递归
               // ... (打印 "if子句" 和 "else子句") ...
               displayAST(node->child->next->next, indent + 2); // 手动递归
               return; // 手动处理子节点，返回
           case NODE_WHILE_STMT:
               // ... (类似地，手动打印 "条件" 和 "循环体") ...
               return;
   ```

3. 操作符

   ```c
           // 操作符
           case NODE_ASSIGN: printf("=\n"); break;
           case NODE_PLUS:   printf("+\n"); break;
           case NODE_MINUS:  printf("-\n"); break;
           case NODE_STAR:   printf("*\n"); break;
           case NODE_DIV:    printf("/\n"); break;
           case NODE_EQ:     printf("==\n"); break;
           case NODE_NE:     printf("!=\n"); break;
           case NODE_GT:     printf(">\n"); break;
           case NODE_GE:     printf(">=\n"); break;
           case NODE_LT:     printf("<\n"); break;
           case NODE_LE:     printf("<=\n"); break;
           case NODE_UMINUS: printf("单目-\n"); break;



```c
    // 递归遍历子结点
    struct ASTNode *child = node->child;
    while (child != NULL) {
        displayAST(child, indent + 1);
        child = child->next;
    }
```

写给break相关的case语句，如操作数，仍需继续执行孩子节点

举例：

当 `displayAST` 处理 `NODE_PLUS` 时：

1. 它打印缩进。
2. `switch` 匹配到 `NODE_PLUS`，打印 `+\n`。
3. 执行 `break;`，跳出 `switch`。
4. 代码来到**这个 `while` 循环**。
5. 循环启动，自动找到 `NODE_PLUS` 的第一个孩子（左边的 `Exp`），并用 `indent + 1` 递归调用 `displayAST`。
6. 循环继续，找到第二个孩子（右边的 `Exp`），并用 `indent + 1` 递归调用 `displayAST`。
7. 循环结束，函数返回。



