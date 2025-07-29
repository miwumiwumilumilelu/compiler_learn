# lab1

## 1. 思考题

### 1.1 正则表达式

1.  **`\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*` 正则表达式匹配的字符串的含义是什么**

   * \w+
     * 匹配所有数字|字母|下划线 多次
   * ([-+.]\w+)*
     * **-|+|.   加上 多次数字|字母|下划线**   匹配0次或多次
   * @
     * 匹配@
   * \w+
   * ([-.]\w+)*
     * 同上，- or .
   * \ .
     * 匹配转义字符 .
   * \w+
   * ([-.]\w+)*

   由于@无特殊含义，所以不需要转义符号；相反就是\ . 匹配的是. 而不是除换行外的任意字符

   

   * **组合：**

     \w+  ——> zjy531725

     ([-+.]\w+)* ——> 出现0次

     @

     \w+ ——> proton

     \ .

     \w+ ——> me

     ([-.]\w+)* ——> 出现0次

     组合上面的: zjy531725@proton.me

     特殊邮箱如 @hust.edu.cn 也会被匹配到

     许多邮箱服务（如 Gmail、Outlook）支持在用户名中使用 `+`，用于子地址，如user+filter@gmail.com 也会被匹配到

     

   * 答案：这个正则表达式用于匹配 **电子邮件地址（Email）**

   

2. **匹配 HTML 注释：编写一个正则表达式，可以匹配 HTML 中的注释，例如 `<!-- This is a comment -->`。**

   * 拆分：

     * <!--			**<!--**
       * This is a comment		**`[\s\S]\*?`**
         * -->			**-->**

   * 细节:

     * 不用. 是因为匹配不了换行符

       <!-- 第一行\n第二行 --> 则匹配不上

     * ? 匹配 0 or 1次

       因为是匹配所有字符，所以防止出现<div>Hello</div>     <div>World</div> 这种情况，匹配了整个字符串

       而不是分别匹配两个,<div>.*?</div> 则分别匹配两次

       *? 让正则引擎 **匹配到第一个满足条件的位置就停止**

     

   * 答案:

     * <!--[\s\S]\*?-->

### 1.2 Flex

1. **如果存在同时以下规则和动作，对于字符串 `+=`，哪条规则会被触发，并尝试解释理由。**

   ```c++
   %%
   \+ { return ADD; }
   = { return ASSIGN; }
   \+= { return ASSIGNADD; }
   %%
   ```

   **\ += { return ASSIGNADD; }**会被触发

   **最长匹配原则**：

   ​	Flex 在匹配时会优先选择 **能匹配最长字符串的规则**

   

   验证程序：

   ```c
   %{
   #include <stdio.h>
   %}
   
   %%
   \+     { printf("ADD\n"); }
   =     { printf("ASSIGN\n"); }
   \+=   { printf("ASSIGNADD\n"); }
   %%
   
   int main() {
       yylex();
       return 0;
   }
   
   int yywrap() {
       return 1;
   }
   ```

   flex xxx.l

   gcc lex.yy.c -o xxx

   ./xxx

   

2. **如果存在同时以下规则和动作，对于字符串 `ABC`，哪条规则会被触发，并尝试解释理由。**

   ```c++
   %%
   ABC { return 1; }
   [a-zA-Z]+ {return 2; }
   %%
   ```

   **ABC { return 1; }**会被触发

   **精确匹配优先**

   ​	`ABC`是 `[a-zA-Z]+`的一个特例，Flex 会优先匹配 **更具体的规则**

   **顺序**

   ​	如果两条规则能匹配同一字符串，**先定义的规则优先**

   

   验证程序：

   ```c
   %{
   #include <stdio.h>
   %}
   
   %%
   ABC        { printf("Rule ABC: 1\n"); }
   [a-zA-Z]+ { printf("Rule [a-zA-Z]+: 2\n"); }
   %%
   
   int main() {
       yylex();
       return 0;
   }
   
   int yywrap() {
       return 1;
   }
   
   ```

   

   

3. **如果存在同时以下规则和动作，对于字符串 `ABC`，哪条规则会被触发，并尝试解释理由。**

   ```c++
   %%
   [a-zA-Z]+ {return 2; }
   ABC { return 1; }
   %%
   ```

   **[a-zA-Z]+ { return 2; }**会被触发
   
   **顺序**
   
   
   
   验证程序:
   
   ```c
   %{
   #include <stdio.h>
   %}
   
   %%
   [a-zA-Z]+ { printf("Rule [a-zA-Z]+: 2\n"); }
   ABC       { printf("Rule ABC: 1\n"); }
   %%
   
   int main() {
       yylex();
       return 0;
   }
   
   int yywrap() {
       return 1;
   }
   
   ```
   
   

### 1.3 Bison

```c
/* calc.y */
%{
#include <stdio.h>
    int yylex(void);
    void yyerror(const char *s);
%}

%token RET
%token <num> NUMBER
%token <op> ADDOP MULOP LPAREN RPAREN
%type <num> top line expr term factor

%start top

%union {
    char   op;
    double num;
}

%%
 //最终结果的合并：top = top + line , 新结果 = 旧结果 + 新行的结果
top
: top line {}
| {}
 //新行的计算：line = expr + "\n" , 新行 = 表达式结果 + 换行符
line
: expr RET
{
    printf(" = %f\n", $1);
}
//表达式计算：1.expt = term 2. expr = expr +- term 
//加减法
expr
: term
{
    $$ = $1;
}
| expr ADDOP term
{
    switch ($2) {
    case '+': $$ = $1 + $3; break;
    case '-': $$ = $1 - $3; break;
    }
}
 //乘除法
 //1.term = factor 2.term = term */ factor
 //越在后面优先级越高 加减<乘除<括号
term
: factor
{
    $$ = $1;
}
| term MULOP factor
{
    switch ($2) {
    case '*': $$ = $1 * $3; break;
    case '/': $$ = $1 / $3; break; // 这里会出什么问题？
    }
}
 //括号内计算，最高优先级
 //1.factor = (expr) 2.factor = (number)
factor
: LPAREN expr RPAREN
{
    $$ = $2;
}
| NUMBER
{
    $$ = $1;
}

%%

void yyerror(const char *s)
{
    fprintf(stderr, "%s\n", s);
}

int main()
{
    yyparse();
    return 0;
}
```

1. **上述计算器例子的文法中存在左递归，为什么 `bison` 可以处理？**

   

   因为它使用了自底向上的 **LR 解析算法**，而这类算法天然支持左递归文法

   

   **左递归:**

   ```c
   expr : expr '+' term  // 左递归：expr 在开头
        | term
   ```

   - **特点**：递归符号（`expr`）出现在规则的 **最左侧**
   - **示例**：`1 + 2 + 3`会被解析为 `(1 + 2) + 3`（左结合）

   **右递归:**

   ```c
   expr : term '+' expr  // 右递归：expr 在末尾
        | term
   ```

   - **特点**：递归符号（`expr`）出现在规则的 **最右侧**
   - **示例**：`1 + 2 + 3`会被解析为 `1 + (2 + 3)`（右结合）

   

   **左递归的优势**:

   - **更自然的结合性**：
     - `1 + 2 + 3`会被解析为 `(1 + 2) + 3`（符合数学习惯）
   - **更高效的解析**：
     - 左递归的 LR 解析只需 **O(n)** 时间和空间（n 是输入长度）
     - 右递归需要 **O(n) 栈空间**，可能导致栈溢出

   

2. **能否修改计算器例子的文法，使得它支持除数 0 规避功能？**

   ```c
   term
   : factor
   {
       $$ = $1;
   }
   | term MULOP factor
   {
       switch ($2) {
           case '*': $$= $1 * $3; break;
           case '/': 
               if ($3 == 0) {
                   yyerror("Error: Division by zero!");
                   exit(1);  // 直接终止程序（或改用 YYERROR 恢复）
               } else {
                   $$ = $1 / $3;
               }
               break;
       }
   }
   ```





## 2. 实验题

lab1目录

> |-- CMakeLists.txt
> |-- build # 在编译过程中产生，不需要通过 git add 添加到仓库中
> |-- src
> |   |-- CMakeLists.txt
> |   |-- common
> |   -- parser
> |       |-- CMakeLists.txt
> |       |-- lexer.c
> |       |-- lexical_analyzer.l # 你需要修改本文件
> |       |-- parser.c
> |       -- syntax_analyzer.y # 你需要修改本文件
> -- tests
>     |-- 1-parser
>     |   |-- input # 针对 Lab1 的测试样例
>     |   |-- output_standard # 助教提供的标准参考结果
>     |   |-- output_student # 测试脚本产生的你解析后的结果
>     |   |-- cleanup.sh
>     |   -- eval_lab1.sh # 测试用的脚本
>     -- testcases_general # 整个课程所用到的测试样例



### 2.1 lexical_analyzer.l

(Cminusf 则是在 Cminus 上追加了浮点操作,Cminus 是 C 语言的一个子集)

**Cminusf 的词法:**

0. 基础四则运算和比较

   ```c
    /* to do for students */
    /* two cases for you, pass_node will send flex's token to bison */
   \+ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return ADD;}
   . { pos_start = pos_end; pos_end++; return ERROR; }
   
    /****请在此补全所有flex的模式与动作  end******/
    /*基础运算*/
   \- {pos_start = pos_end; pos_end += 1; pass_node(yytext); return SUB;}
   \* {pos_start = pos_end; pos_end += 1; pass_node(yytext); return MUL;}
   \/ {pos_start = pos_end; pos_end += 1; pass_node(yytext);return DIV;}
   \< {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LT;}
   \> {pos_start = pos_end; pos_end += 1; pass_node(yytext); return GT;}
   \= {pos_start = pos_end; pos_end += 1; pass_node(yytext); return ASSIN;}
   ">=" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return GTE;}
   "<=" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return LTE;}
   "==" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return EQ;} 
   "!=" {pos_start = pos_end; pos_end +=2 ; pass_node(yytext); return NEQ;}
   ```



1. 关键字

   ```
   else if int return void while float
   ```

   ```c
    /*关键字*/
   else {pos_start = pos_end; pos_end += 4; pass_node(yytext); return ELSE;}
   if {pos_start = pos_end; pos_end += 2; pass_node(yytext); return IF;}
   int {pos_start = pos_end; pos_end += 3; pass_node(yytext); return INT;}
   return {pos_start = pos_end; pos_end += 6; pass_node(yytext); return RETURN;}
   void {pos_start = pos_end; pos_end += 4; pass_node(yytext); return VOID;}
   while {pos_start = pos_end; pos_end += 5; pass_node(yytext); return WHILE;}
   float {pos_start = pos_end; pos_end += 5; pass_node(yytext); return FLOAT;}
   ```

   

2. 专用符号

   ```
   + - * / < <= > >= == != = ; , ( ) [ ] { } /* */
   ```

   ```c
    /*符号,注释后面单独写*/
   \; {pos_start = pos_end; pos_end += 1; pass_node(yytext); return SEMICOLON;}
   \, {pos_start = pos_end; pos_end += 1; pass_node(yytext); return COMMA;}
   \( {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LPARENTHESE;}
   \) {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RPARENTHESE;}
   \[ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACKET;}
   \] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACKET;}
   \{ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACE;}
   \} {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACE;}
   ```

   

3. 标识符 ID 和数值，通过下列正则表达式定义：

   ```
   letter = a|...|z|A|...|Z
   digit = 0|...|9
   ID = letter+
   INTEGER = digit+
   FLOAT = (digit+. | digit*.digit+)
   ```

   ```c
    /*ID & NUM*/
   [a-zA-Z] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LETTER;}
   [0-9] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return DIGIT;}
   [a-zA-Z]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return ID;}
   [0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return INTEGER;}
   [0-9]+\.|[0-9]*\.[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return FLOATPOINT;}
   ```

   

4. 注释用 `/*...*/` 表示，可以超过一行。注释不能嵌套。

   ```
   /*...*/
   ```

   ```c
   %x COMMENT
   /*注释等其他特殊符合*/
   "/*" { 
       BEGIN(COMMENT);
       pos_end += 2;
   }
   <COMMENT>{
       "*/" { 
           BEGIN(INITIAL);
           pos_end += 2;
       }
       \n { 
           lines++;
           pos_end = 1;
       }    
       . {pos_end += 1;}
   }
   \n  {lines++ ; pos_start = 1; pos_end = 1;}
   \[\] {pos_start = pos_end; pos_end += 2; pass_node(yytext);return ARRAY;}
   [ \f\r\t\v] {pos_start = pos_end;pos_end += strlen(yytext);}
   ```

Flex 词法分析器中，使用 `[ \f\n\r\t\v]`显式枚举空白符，而非直接使用 `\s`

是因为**Flex 的默认正则引擎不完全支持 `\s`**：

- Flex 基于传统的 POSIX 正则，原生不支持 `\s`、`\d`等快捷符号
- 若强行使用 `\s`，需开启扩展模式（如 `%option reentrant`或 `%option bison-bridge`），但会增加复杂性



词法特性相比 C 语言做了大量简化，比如标识符 `student_id` 在 C 语言中是合法的，但是在 Cminusf 中是不合法的

```c
%x COMMENT
%%
 /* to do for students */
 /* two cases for you, pass_node will send flex's token to bison */
\+ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return ADD;}
. { pos_start = pos_end; pos_end++; return ERROR; }

 /****请在此补全所有flex的模式与动作  end******/
\- {pos_start = pos_end; pos_end += 1; pass_node(yytext); return SUB;}
\* {pos_start = pos_end; pos_end += 1; pass_node(yytext); return MUL;}
\/ {pos_start = pos_end; pos_end += 1; pass_node(yytext);return DIV;}
\< {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LT;}
\> {pos_start = pos_end; pos_end += 1; pass_node(yytext); return GT;}
\= {pos_start = pos_end; pos_end += 1; pass_node(yytext); return ASSIN;}
">=" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return GTE;}
"<=" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return LTE;}
"==" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return EQ;} 
"!=" {pos_start = pos_end; pos_end +=2 ; pass_node(yytext); return NEQ;}

\; {pos_start = pos_end; pos_end += 1; pass_node(yytext); return SEMICOLON;}
\, {pos_start = pos_end; pos_end += 1; pass_node(yytext); return COMMA;}
\( {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LPARENTHESE;}
\) {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RPARENTHESE;}
\[ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACKET;}
\] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACKET;}
\{ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACE;}
\} {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACE;}

else {pos_start = pos_end; pos_end += 4; pass_node(yytext); return ELSE;}
if {pos_start = pos_end; pos_end += 2; pass_node(yytext); return IF;}
int {pos_start = pos_end; pos_end += 3; pass_node(yytext); return INT;}
return {pos_start = pos_end; pos_end += 6; pass_node(yytext); return RETURN;}
void {pos_start = pos_end; pos_end += 4; pass_node(yytext); return VOID;}
while {pos_start = pos_end; pos_end += 5; pass_node(yytext); return WHILE;}
float {pos_start = pos_end; pos_end += 5; pass_node(yytext); return FLOAT;}

[a-zA-Z] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LETTER;}
[0-9] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return DIGIT;}
[a-zA-Z]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return ID;}
[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return INTEGER;}
[0-9]+\.|[0-9]*\.[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return FLOATPOINT;}

"/*" { 
    BEGIN(COMMENT);
    pos_end += 2;
}
<COMMENT>{
    "*/" { 
        BEGIN(INITIAL);
        pos_end += 2;
    }
    \n { 
        lines++;
        pos_end = 1;
    }    
    . {pos_end += 1;}
}
\n  {lines++ ; pos_start = 1; pos_end = 1;}
\[\] {pos_start = pos_end; pos_end += 2; pass_node(yytext);return ARRAY;}
[ \f\r\t\v] {pos_start = pos_end;pos_end += strlen(yytext);}
%%
```



### 2.2 syntax_analyzer.y

**Cminusf 的语法:**

我们将 Cminusf 的所有规则分为五类：

1. 字面量、关键字、运算符与标识符
   - `type-specifier`
   - `relop`
   - `addop`
   - `mulop`
2. 声明
   - `declaration-list`
   - `declaration`
   - `var-declaration`
   - `fun-declaration`
   - `local-declarations`
3. 语句
   - `compound-stmt`
   - `statement-list`
   - `statement`
   - `expression-stmt`
   - `iteration-stmt`
   - `selection-stmt`
   - `return-stmt`
4. 表达式
   - `expression`
   - `simple-expression`
   - `var`
   - `additive-expression`
   - `term`
   - `factor`
   - `integer`
   - `float`
   - `call`
5. 其他
   - `params`
   - `param-list`
   - `param`
   - `args`
   - `arg-list`



```c
%token <node> ERROR
%token <node> ADD SUB MUL DIV
%token <node> LT LTE GT GTE EQ NEQ ASSIN
%token <node> SEMICOLON COMMA LPARENTHESE RPARENTHESE LBRACKET RBRACKET LBRACE RBRACE
%token <node> ELSE IF INT FLOAT RETURN VOID WHILE ID LETTER DIGIT INTEGER FLOATPOINT ARRAY
%type <node> type-specifier relop addop mulop
%type <node> declaration-list declaration var-declaration fun-declaration local-declarations
%type <node> compound-stmt statement-list statement expression-stmt iteration-stmt selection-stmt return-stmt
%type <node> simple-expression expression var additive-expression term factor integer float call
%type <node> params param-list param args arg-list
%type <node> program

%start program
```



1. program→declaration-list
2. declaration-list→declaration-list declaration ∣ declarationdeclaration-list→declaration-list declaration ∣ declaration
3. declaration→var-declaration ∣ fun-declarationdeclaration→var-declaration ∣ fun-declaration
4. var-declaration →type-specifier ID‾ ;‾ ∣ type-specifier ID‾ [‾ INTEGER‾ ]‾ ;‾var-declaration →type-specifier **ID** **;** ∣ type-specifier **ID** **[** **INTEGER** **]** **;**
5. type-specifier→int‾ ∣ float‾ ∣ void‾type-specifier→**int** ∣ **float** ∣ **void**
6. fun-declaration→type-specifier ID‾ (‾ params )‾ compound-stmtfun-declaration→type-specifier **ID** **(** params **)** compound-stmt
7. params→param-list ∣ void‾params→param-list ∣ **void**
8. param-list→param-list ,‾ param ∣ paramparam-list→param-list **,** param ∣ param
9. param→type-specifier ID‾ ∣ type-specifier ID‾ [‾ ]‾param→type-specifier **ID** ∣ type-specifier **ID** **[** **]**
10. compound-stmt→{‾ local-declarations statement-list }‾compound-stmt→**{** local-declarations statement-list **}**
11. local-declarations→local-declarations var-declaration ∣ emptylocal-declarations→local-declarations var-declaration ∣ empty
12. statement-list→statement-list statement ∣ emptystatement-list→statement-list statement ∣ empty
13. statement→ expression-stmt∣ compound-stmt∣ selection-stmt∣ iteration-stmt∣ return-stmtstatement→ expression-stmt∣ compound-stmt∣ selection-stmt∣ iteration-stmt∣ return-stmt
14. expression-stmt→expression ;‾ ∣ ;‾expression-stmt→expression **;** ∣ **;**
15. selection-stmt→ if‾ (‾ expression )‾ statement∣ if‾ (‾ expression )‾ statement else‾ statementselection-stmt→ **if** **(** expression **)** statement∣ **if** **(** expression **)** statement **else** statement
16. iteration-stmt→while‾ (‾ expression )‾ statementiteration-stmt→**while** **(** expression **)** statement
17. return-stmt→return‾ ;‾ ∣ return‾ expression ;‾return-stmt→**return** **;** ∣ **return** expression **;**
18. expression→var =‾ expression ∣ simple-expressionexpression→var **=** expression ∣ simple-expression
19. var→ID‾ ∣ ID‾ [‾ expression]‾var→**ID** ∣ **ID** **[** expression**]**
20. simple-expression→additive-expression relop additive-expression ∣ additive-expressionsimple-expression→additive-expression relop additive-expression ∣ additive-expression
21. relop →<=‾ ∣ <‾ ∣ >‾ ∣ >=‾ ∣ ==‾ ∣ !=‾relop →**<=** ∣ **<** ∣ **>** ∣ **>=** ∣ **==** ∣ **!=**
22. additive-expression→additive-expression addop term ∣ termadditive-expression→additive-expression addop term ∣ term
23. addop→+‾ ∣ -‾addop→**+** ∣ **-**
24. term→term mulop factor ∣ factorterm→term mulop factor ∣ factor
25. mulop→*‾ ∣ /‾mulop→***** ∣ **/**
26. factor→(‾ expression )‾ ∣ var ∣ call ∣ integer ∣ floatfactor→**(** expression **)** ∣ var ∣ call ∣ integer ∣ float
27. integer→INTEGER‾integer→**INTEGER**
28. float→FLOATPOINT‾float→**FLOATPOINT**
29. call→ID‾ (‾ args)‾call→**ID** **(** args**)**
30. args→arg-list ∣ emptyargs→arg-list ∣ empty
31. arg-list→arg-list ,‾ expression ∣ expressionarg-list→arg-list **,** expression ∣ expression

```c
program : declaration-list { $$ = node("program", 1, $1); gt->root = $$; }
declaration-list : declaration-list declaration { $$ = node("declaration-list", 2, $1, $2);}
                 | declaration { $$ = node("declaration-list", 1, $1); }
declaration : var-declaration { $$ = node("declaration", 1, $1); }
            | fun-declaration { $$ = node("declaration", 1, $1); }
var-declaration : type-specifier ID SEMICOLON { $$ = node("var-declaration", 3, $1, $2, $3); }
                | type-specifier ID LBRACKET INTEGER RBRACKET SEMICOLON { $$ = node("var-declaration", 6, $1, $2, $3, $4, $5, $6); }
type-specifier : INT { $$ = node("type-specifier", 1, $1); }
               | FLOAT { $$ = node("type-specifier", 1, $1); }
               | VOID { $$ = node("type-specifier", 1, $1); }
fun-declaration : type-specifier ID LPARENTHESE params RPARENTHESE compound-stmt { $$ = node("fun-declaration", 6, $1, $2, $3, $4, $5, $6); }
params : param-list { $$ = node("params", 1, $1); }
       | VOID { $$ = node("params", 1, $1); }
param-list : param-list COMMA param { $$ = node("param-list", 3, $1, $2, $3); }
           | param { $$ = node("param-list", 1, $1); }
param : type-specifier ID { $$ = node("param", 2, $1, $2); }
      | type-specifier ID ARRAY { $$ = node("param", 3, $1, $2, $3); }
compound-stmt : LBRACE local-declarations statement-list RBRACE { $$ = node("compound-stmt", 4, $1, $2, $3, $4); }
local-declarations : { $$ = node("local-declarations", 0); }
                   | local-declarations var-declaration { $$ = node("local-declarations", 2, $1, $2); }
statement-list : { $$ = node("statement-list", 0); }
               | statement-list statement { $$ = node("statement-list", 2, $1, $2); }
statement : expression-stmt { $$ = node("statement", 1, $1); }
          | compound-stmt { $$ = node("statement", 1, $1); }
          | selection-stmt { $$ = node("statement", 1, $1); }
          | iteration-stmt { $$ = node("statement", 1, $1); }
          | return-stmt { $$ = node("statement", 1, $1); }
expression-stmt : expression SEMICOLON { $$ = node("expression-stmt", 2, $1, $2); }
                | SEMICOLON { $$ = node("expression-stmt", 1, $1); }
selection-stmt : IF LPARENTHESE expression RPARENTHESE statement { $$ = node("selection-stmt", 5, $1, $2, $3, $4, $5); }
               | IF LPARENTHESE expression RPARENTHESE statement ELSE statement { $$ = node("selection-stmt", 7, $1, $2, $3, $4, $5, $6, $7); }
iteration-stmt : WHILE LPARENTHESE expression RPARENTHESE statement { $$ = node("iteration-stmt", 5, $1, $2, $3, $4, $5); }

return-stmt : RETURN SEMICOLON { $$ = node("return-stmt", 2, $1, $2); }
            | RETURN expression SEMICOLON { $$ = node("return-stmt", 3, $1, $2, $3); }
expression : var ASSIN expression { $$ = node("expression", 3, $1, $2, $3); }
           | simple-expression { $$ = node("expression", 1, $1); }
var : ID { $$ = node("var", 1, $1); }
    | ID LBRACKET expression RBRACKET { $$ = node("var", 4, $1, $2, $3, $4); }
simple-expression : additive-expression relop additive-expression { $$ = node("simple-expression", 3, $1, $2, $3); }
                  | additive-expression { $$ = node("simple-expression", 1, $1); }
relop : LTE { $$ = node("relop", 1, $1); }
      | LT { $$ = node("relop", 1, $1); }
      | GT { $$ = node("relop", 1, $1); }
      | GTE { $$ = node("relop", 1, $1); }
      | EQ { $$ = node("relop", 1, $1); }
      | NEQ { $$ = node("relop", 1, $1); }
additive-expression : additive-expression addop term { $$ = node("additive-expression", 3, $1, $2, $3); }
                    | term { $$ = node("additive-expression", 1, $1); }
addop : ADD { $$ = node("addop", 1, $1); }
      | SUB { $$ = node("addop", 1, $1); }
term : term mulop factor { $$ = node("term", 3, $1, $2, $3); }
     | factor { $$ = node("term", 1, $1); }
mulop : MUL { $$ = node("mulop", 1, $1); }
      | DIV { $$ = node("mulop", 1, $1); }
factor : LPARENTHESE expression RPARENTHESE { $$ = node("factor", 3, $1, $2, $3); }
       | var { $$ = node("factor", 1, $1); }
       | call { $$ = node("factor", 1, $1); }
       | integer { $$ = node("factor", 1, $1); }
       | float { $$ = node("factor", 1, $1); }
integer : INTEGER { $$ = node("integer", 1, $1); }
float : FLOATPOINT { $$ = node("float", 1, $1); }
call : ID LPARENTHESE args RPARENTHESE { $$ = node("call", 4, $1, $2, $3, $4); }
args : { $$ = node("args", 0); }
     | arg-list { $$ = node("args", 1, $1); }
arg-list : arg-list COMMA expression { $$ = node("arg-list", 3, $1, $2, $3); }
         | expression { $$ = node("arg-list", 1, $1); }
```



### 2.3 遇到的问题与解决方案

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1$ lexer tests/testcases_general/1-return.cminus
Token         Text      Line    Column (Start,End)
283           void      0       (0,4)
258                     0       (4,5)
285           main      0       (5,9)
258              (      0       (9,10)
283           void      0       (10,14)
258              )      0       (14,15)
258                     0       (15,16)
258              {      0       (16,17)
258                     0       (17,18)
282         return      0       (18,24)
258              ;      0       (24,25)
258                     0       (25,26)
258              }      0       (26,27)
```

**问题1：忘记初始化**

```c
int lines = 1;
int pos_start = 1;
int pos_end = 1;
```



**关键!!!**

**问题2：注意冲突字符的规则顺序**

由于是顺序识别规则：

所以**虽然**规则 `[ \f\r\t\v]`用于匹配空白字符（不返回 token）,**但是**`.`规则（匹配任何字符）被错误地放置在更靠前的位置，导致空白字符被 `.`规则捕获，返回 `ERROR`token（258）

将`.`规则放在最后

```c
manbin@compile:~/2023_warm_up_b/_lab1/lab1$ lexer tests/testcases_general/1-return.cminus
Token         Text      Line    Column (Start,End)
283           void      1       (1,5)
285           main      1       (6,10)
272              (      1       (10,11)
283           void      1       (11,15)
273              )      1       (15,16)
276              {      1       (17,18)
282         return      1       (19,25)
270              ;      1       (25,26)
277              }      1       (27,28)
```



**问题3：注意冲突字符的规则顺序**

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1$ cat ./tests/testcases_general/2-decl_int.cminus 
void main(void) {
    int a;
    return;
}
manbin@compile:~/2023_warm_up_b/_lab1/lab1$ parser ./tests/testcases_general/2-decl_int.cminus 
error at line 2 column 9: syntax error
```

语法分析器在处理时 `type-specifier ID`，测试文件中的 `a`被识别为 `LETTER`而不是 `ID`

- 规则 `[a-zA-Z]`会匹配单个字母，而 `[a-zA-Z]+`会匹配多个字母
- 但 `[a-zA-Z]`规则在 `[a-zA-Z]+`之前，所以单个字母会被匹配为 `LETTER`而不是 `ID`

**调整顺序:**

```c
 /*ID & NUM*/
[a-zA-Z]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return ID;}
[a-zA-Z] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LETTER;}
[0-9] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return DIGIT;}
[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return INTEGER;}
[0-9]+\.|[0-9]*\.[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return FLOATPOINT;}
```



**问题4：很多文件第一行解析就出错了，是因为单行注释规则没写**

添加

```c
"//"[^\n]* {pos_start = pos_end;pos_end += strlen(yytext);}
```



**问题5：注意冲突字符的规则顺序**

修改`yyerror`，便于调试

```c
fprintf(stderr, "[ERR]: unable to analysize %s at %d line, from %d to %d: %s\n", yytext, lines, pos_start, pos_end, s);
```

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1/tests/1-parser/input/easy$ parser FAIL_comment2.cminus 
[ERR]: unable to analysize 0 at 4 line, from 12 to 13: syntax error
manbin@compile:~/2023_warm_up_b/_lab1/lab1/tests/1-parser/input/easy$ lexer FAIL_comment2.cminus 
Token         Text      Line    Column (Start,End)
280            int      3       (1,4)
285           main      3       (5,9)
272              (      3       (9,10)
283           void      3       (10,14)
273              )      3       (14,15)
276              {      3       (15,16)
282         return      4       (5,11)
287              0      4       (12,13)
270              ;      4       (13,14)
277              }      5       (1,2)
manbin@compile:~/2023_warm_up_b/_lab1/lab1/tests/1-parser/input/easy$ cat FAIL_comment2.cminus 
// cminus dont support comment like that

int main(void){
    return 0;
}
```

```c
// 语法规则相关部分
return-stmt : RETURN SEMICOLON
            | RETURN expression SEMICOLON

expression : simple-expression  // 进入这里

simple-expression : additive-expression

additive-expression : term

term : factor

factor : integer
       | ...

integer : INTEGER  // 应该匹配这里 <<<< 但解析失败
```

**问题关键**：解析器在 `RETURN`后遇到 `0`时，应该匹配 `integer → factor → term → ...`路径但失败了

同样是顺序问题，**INTEGER**

```c
 /*ID & NUM*/
[a-zA-Z]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return ID;}
[a-zA-Z] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LETTER;}
[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return INTEGER;}
[0-9] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return DIGIT;}
[0-9]+\.|[0-9]*\.[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return FLOATPOINT;}
```



**至此easy通关**

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1/tests/1-parser$ ./eval_lab1.sh easy yes
[info] Analyzing expr.cminus
[info] Analyzing FAIL_comment2.cminus
[ERR]: unable to analysize / at 1 line, from 1 to 2: syntax error
[info] Analyzing FAIL_comment.cminus
[ERR]: unable to analysize  at 1 line, from 1 to 20: syntax error
[info] Analyzing FAIL_function.cminus
[ERR]: unable to analysize  at 3 line, from 1 to 2: syntax error
[info] Analyzing FAIL_id.cminus
[ERR]: unable to analysize 1 at 1 line, from 6 to 7: syntax error
[info] Analyzing id.cminus
[info] Comparing...
[info] No difference! Congratulations!
```

————————————————————————————————————————————————————————————

**问题6：测试normal遇到了唯一一个错误——[ ]和ARRAY**

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1/tests/1-parser$ ./eval_lab1.sh normal yes
[info] Analyzing array.cminus
[info] Analyzing FAIL_assign.cminus
[ERR]: unable to analysize = at 4 line, from 4 to 5: syntax error
[info] Analyzing FAIL_local-decl.cminus
[ERR]: unable to analysize int at 4 line, from 5 to 8: syntax error
[info] Analyzing func.cminus
[info] Analyzing if.cminus
[info] Analyzing local-decl.cminus
[info] Analyzing skip_spaces.cminus
[info] Comparing...
Files /home/manbin/2023_warm_up_b/_lab1/lab1/tests/1-parser/output_student/normal/func.syntax_tree and /home/manbin/2023_warm_up_b/_lab1/lab1/tests/1-parser/output_standard/normal/func.syntax_tree differ
```

**查看语法树区别：**

我的：

```c
|  |  |  |  |  |  |  |  >--* []
```

标准的：

```c
|  |  |  |  |  |  |  |  >--* [
|  |  |  |  |  |  |  |  >--* ]
```

**查看源文件：**

```c
float foo(float a, float b[]) {
	return 1;
}

int main(void) {
	return 0;
}
```

对于【】由于最长优先匹配，确实匹配到了ARRAY，但是标准不是这样的，所以删了ARRAY规则，然后修改相应出现的语法规则位置，改为`LBRACKET RBRACKET`

```c
param : type-specifier ID { $$ = node("param", 2, $1, $2); }
      | type-specifier ID LBRACKET RBRACKET { $$ = node("param", 4, $1, $2, $3, $4); }
```

**至此normal通关**

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1/tests/1-parser$ ./eval_lab1.sh normal yes
[info] Analyzing array.cminus
[info] Analyzing FAIL_assign.cminus
[ERR]: unable to analysize = at 4 line, from 4 to 5: syntax error
[info] Analyzing FAIL_local-decl.cminus
[ERR]: unable to analysize int at 4 line, from 5 to 8: syntax error
[info] Analyzing func.cminus
[info] Analyzing if.cminus
[info] Analyzing local-decl.cminus
[info] Analyzing skip_spaces.cminus
[info] Comparing...
[info] No difference! Congratulations!
```

————————————————————————————————————————————————————————————

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1/tests/1-parser$ ./eval_lab1.sh hard yes
[info] Analyzing assoc.cminus
[info] Analyzing gcd.cminus
[info] Analyzing hanoi.cminus
[info] Analyzing if.cminus
[info] Analyzing selectionsort.cminus
[info] Analyzing You_Should_Pass.cminus
[info] Comparing...
[info] No difference! Congratulations!
```

**至此hard也通关，全部测试完成**

————————————————————————————————————————————————————————————



### **2.4 最终答案**

```c
%option noyywrap
%x COMMENT
%{
/*****************声明和选项设置  begin*****************/
#include <stdio.h>
#include <stdlib.h>

#include "../include/common/syntax_tree.h"
#include "syntax_analyzer.h"

int lines = 1;
int pos_start = 1;
int pos_end = 1;

void pass_node(char *text){
     yylval.node = new_syntax_tree_node(text);
}

/*****************声明和选项设置  end*****************/

%}


%%
 /* to do for students */
 /* two cases for you, pass_node will send flex's token to bison */
\+ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return ADD;}

 /****请在此补全所有flex的模式与动作  end******/
 /*基础运算*/
\- {pos_start = pos_end; pos_end += 1; pass_node(yytext); return SUB;}
\* {pos_start = pos_end; pos_end += 1; pass_node(yytext); return MUL;}
\/ {pos_start = pos_end; pos_end += 1; pass_node(yytext);return DIV;}
\< {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LT;}
\> {pos_start = pos_end; pos_end += 1; pass_node(yytext); return GT;}
\= {pos_start = pos_end; pos_end += 1; pass_node(yytext); return ASSIN;}
">=" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return GTE;}
"<=" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return LTE;}
"==" {pos_start = pos_end; pos_end += 2; pass_node(yytext); return EQ;} 
"!=" {pos_start = pos_end; pos_end +=2 ; pass_node(yytext); return NEQ;}

 /*符号*/
\; {pos_start = pos_end; pos_end += 1; pass_node(yytext); return SEMICOLON;}
\, {pos_start = pos_end; pos_end += 1; pass_node(yytext); return COMMA;}
\( {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LPARENTHESE;}
\) {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RPARENTHESE;}
\[ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACKET;}
\] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACKET;}
\{ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACE;}
\} {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACE;}

 /*关键字*/
else {pos_start = pos_end; pos_end += 4; pass_node(yytext); return ELSE;}
if {pos_start = pos_end; pos_end += 2; pass_node(yytext); return IF;}
int {pos_start = pos_end; pos_end += 3; pass_node(yytext); return INT;}
return {pos_start = pos_end; pos_end += 6; pass_node(yytext); return RETURN;}
void {pos_start = pos_end; pos_end += 4; pass_node(yytext); return VOID;}
while {pos_start = pos_end; pos_end += 5; pass_node(yytext); return WHILE;}
float {pos_start = pos_end; pos_end += 5; pass_node(yytext); return FLOAT;}

 /*ID & NUM*/
[a-zA-Z]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return ID;}
[a-zA-Z] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LETTER;}
[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return INTEGER;}
[0-9] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return DIGIT;}
[0-9]+\.|[0-9]*\.[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return FLOATPOINT;}

 /*注释等其他特殊符号*/
"/*" { 
    BEGIN(COMMENT);
    pos_end += 2;
}
<COMMENT>{
    "*/" { 
        BEGIN(INITIAL);
        pos_end += 2;
    }
    \n { 
        lines++;
        pos_end = 1;
    }    
    . {pos_end += 1;}
}
\n  {lines++ ; pos_start = 1; pos_end = 1;}
[ \f\t\r\v] {pos_start = pos_end;pos_end += strlen(yytext);}
. { pos_start = pos_end; pos_end++; return ERROR; }
%%

```

```c
%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "../include/common/syntax_tree.h" 

// external functions from lex
extern int yylex();
extern int yyparse();
extern int yyrestart();
extern FILE * yyin;

// external variables from lexical_analyzer module
extern int lines;
extern char * yytext;
extern int pos_end;
extern int pos_start;

// Global syntax tree
syntax_tree *gt;

// Error reporting
void yyerror(const char *s);

// Helper functions written for you with love
syntax_tree_node *node(const char *node_name, int children_num, ...);
%}

/* TODO: Complete this definition.
   Hint: See pass_node(), node(), and syntax_tree.h.
         Use forward declaring. */
%union {
	struct _syntax_tree_node *node;
}

/* TODO: Your tokens here. */
%token <node> ERROR
%token <node> ADD SUB MUL DIV
%token <node> LT LTE GT GTE EQ NEQ ASSIN
%token <node> SEMICOLON COMMA LPARENTHESE RPARENTHESE LBRACKET RBRACKET LBRACE RBRACE
%token <node> ELSE IF INT FLOAT RETURN VOID WHILE ID LETTER DIGIT INTEGER FLOATPOINT ARRAY
%type <node> type-specifier relop addop mulop
%type <node> declaration-list declaration var-declaration fun-declaration local-declarations
%type <node> compound-stmt statement-list statement expression-stmt iteration-stmt selection-stmt return-stmt
%type <node> simple-expression expression var additive-expression term factor integer float call
%type <node> params param-list param args arg-list
%type <node> program

%start program

%%
/* TODO: Your rules here. */

/* Example:
program: declaration-list {$$ = node( "program", 1, $1); gt->root = $$;}
       ;
*/

program : declaration-list { $$ = node("program", 1, $1); gt->root = $$; }
declaration-list : declaration-list declaration { $$ = node("declaration-list", 2, $1, $2);}
                 | declaration { $$ = node("declaration-list", 1, $1); }
declaration : var-declaration { $$ = node("declaration", 1, $1); }
            | fun-declaration { $$ = node("declaration", 1, $1); }
var-declaration : type-specifier ID SEMICOLON { $$ = node("var-declaration", 3, $1, $2, $3); }
                | type-specifier ID LBRACKET INTEGER RBRACKET SEMICOLON { $$ = node("var-declaration", 6, $1, $2, $3, $4, $5, $6); }
type-specifier : INT { $$ = node("type-specifier", 1, $1); }
               | FLOAT { $$ = node("type-specifier", 1, $1); }
               | VOID { $$ = node("type-specifier", 1, $1); }
fun-declaration : type-specifier ID LPARENTHESE params RPARENTHESE compound-stmt { $$ = node("fun-declaration", 6, $1, $2, $3, $4, $5, $6); }
params : param-list { $$ = node("params", 1, $1); }
       | VOID { $$ = node("params", 1, $1); }
param-list : param-list COMMA param { $$ = node("param-list", 3, $1, $2, $3); }
           | param { $$ = node("param-list", 1, $1); }
param : type-specifier ID { $$ = node("param", 2, $1, $2); }
      | type-specifier ID LBRACKET RBRACKET { $$ = node("param", 4, $1, $2, $3, $4); }
compound-stmt : LBRACE local-declarations statement-list RBRACE { $$ = node("compound-stmt", 4, $1, $2, $3, $4); }
local-declarations : { $$ = node("local-declarations", 0); }
                   | local-declarations var-declaration { $$ = node("local-declarations", 2, $1, $2); }
statement-list : { $$ = node("statement-list", 0); }
               | statement-list statement { $$ = node("statement-list", 2, $1, $2); }
statement : expression-stmt { $$ = node("statement", 1, $1); }
          | compound-stmt { $$ = node("statement", 1, $1); }
          | selection-stmt { $$ = node("statement", 1, $1); }
          | iteration-stmt { $$ = node("statement", 1, $1); }
          | return-stmt { $$ = node("statement", 1, $1); }
expression-stmt : expression SEMICOLON { $$ = node("expression-stmt", 2, $1, $2); }
                | SEMICOLON { $$ = node("expression-stmt", 1, $1); }
selection-stmt : IF LPARENTHESE expression RPARENTHESE statement { $$ = node("selection-stmt", 5, $1, $2, $3, $4, $5); }
               | IF LPARENTHESE expression RPARENTHESE statement ELSE statement { $$ = node("selection-stmt", 7, $1, $2, $3, $4, $5, $6, $7); }
iteration-stmt : WHILE LPARENTHESE expression RPARENTHESE statement { $$ = node("iteration-stmt", 5, $1, $2, $3, $4, $5); }

return-stmt : RETURN SEMICOLON { $$ = node("return-stmt", 2, $1, $2); }
            | RETURN expression SEMICOLON { $$ = node("return-stmt", 3, $1, $2, $3); }
expression : var ASSIN expression { $$ = node("expression", 3, $1, $2, $3); }
           | simple-expression { $$ = node("expression", 1, $1); }
var : ID { $$ = node("var", 1, $1); }
    | ID LBRACKET expression RBRACKET { $$ = node("var", 4, $1, $2, $3, $4); }
simple-expression : additive-expression relop additive-expression { $$ = node("simple-expression", 3, $1, $2, $3); }
                  | additive-expression { $$ = node("simple-expression", 1, $1); }
relop : LTE { $$ = node("relop", 1, $1); }
      | LT { $$ = node("relop", 1, $1); }
      | GT { $$ = node("relop", 1, $1); }
      | GTE { $$ = node("relop", 1, $1); }
      | EQ { $$ = node("relop", 1, $1); }
      | NEQ { $$ = node("relop", 1, $1); }
additive-expression : additive-expression addop term { $$ = node("additive-expression", 3, $1, $2, $3); }
                    | term { $$ = node("additive-expression", 1, $1); }
addop : ADD { $$ = node("addop", 1, $1); }
      | SUB { $$ = node("addop", 1, $1); }
term : term mulop factor { $$ = node("term", 3, $1, $2, $3); }
     | factor { $$ = node("term", 1, $1); }
mulop : MUL { $$ = node("mulop", 1, $1); }
      | DIV { $$ = node("mulop", 1, $1); }
factor : LPARENTHESE expression RPARENTHESE { $$ = node("factor", 3, $1, $2, $3); }
       | var { $$ = node("factor", 1, $1); }
       | call { $$ = node("factor", 1, $1); }
       | integer { $$ = node("factor", 1, $1); }
       | float { $$ = node("factor", 1, $1); }
integer : INTEGER { $$ = node("integer", 1, $1); }
float : FLOATPOINT { $$ = node("float", 1, $1); }
call : ID LPARENTHESE args RPARENTHESE { $$ = node("call", 4, $1, $2, $3, $4); }
args : { $$ = node("args", 0); }
     | arg-list { $$ = node("args", 1, $1); }
arg-list : arg-list COMMA expression { $$ = node("arg-list", 3, $1, $2, $3); }
         | expression { $$ = node("arg-list", 1, $1); }
%%

/// The error reporting function.
void yyerror(const char * s)
{
    // TO STUDENTS: This is just an example.
    // You can customize it as you like.
    fprintf(stderr, "[ERR]: unable to analysize %s at %d line, from %d to %d: %s\n", yytext, lines, pos_start, pos_end, s);
}

/// Parse input from file `input_path`, and prints the parsing results
/// to stdout.  If input_path is NULL, read from stdin.
///
/// This function initializes essential states before running yyparse().
syntax_tree *parse(const char *input_path)
{
    if (input_path != NULL) {
        if (!(yyin = fopen(input_path, "r"))) {
            fprintf(stderr, "[ERR] Open input file %s failed.\n", input_path);
            exit(1);
        }
    } else {
        yyin = stdin;
    }

    lines = pos_start = pos_end = 1;
    gt = new_syntax_tree();
    yyrestart(yyin);
    yyparse();
    return gt;
}

/// A helper function to quickly construct a tree node.
///
/// e.g. $$ = node("program", 1, $1);
syntax_tree_node *node(const char *name, int children_num, ...)
{
    syntax_tree_node *p = new_syntax_tree_node(name);
    syntax_tree_node *child;
    if (children_num == 0) {
        child = new_syntax_tree_node("epsilon");
        syntax_tree_add_child(p, child);
    } else {
        va_list ap;
        va_start(ap, children_num);
        for (int i = 0; i < children_num; ++i) {
            child = va_arg(ap, syntax_tree_node *);
            syntax_tree_add_child(p, child);
        }
        va_end(ap);
    }
    return p;
}

```

### 2.5 总结和思路

**调试错误尤其注意词法规则的顺序性，然后记得看output_standard和output_student的输出语法树的差异，如果standard是空输出，说明存在语法错误，自信点不要轻易否定自己**

先看实验细节与要求，根据cminusf的文法，列出所有token和type并且根据这个顺序可以去写相关.y文件的规则

因为`%token`定义的符号需要与 Flex 词法分析器的返回值匹配，是词法分析器的接口；

​	`%type`声明的非终结符必须有对应的语法规则实现，用来将Token 组合成非终结符节点，这可以看语法分析的实验内容，一一对应补全

所以token需要在.l文件中一一匹配相应规则，此时再看词法分析的实验内容，将其匹配规则和行为补全

