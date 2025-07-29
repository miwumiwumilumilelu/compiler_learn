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
    /*注释等其他特殊符合*/
   \n  {lines++ ; pos_start = 1; pos_end = 1;}
   \/\*([^\*]|\*+[^*/])*\*+\/ {
        for(int i=0;i<strlen(yytext);i++)
        {
   		if(yytext[i]=='\n')
             {
   			pos_start=1;
   			pos_end=1;
   			lines++;
   		}
   		else pos_end++;
        }
   }
   [ \f\n\r\t\v] {pos_start = pos_end;pos_end += strlen(yytext);}
   ```

Flex 词法分析器中，使用 `[ \f\n\r\t\v]`显式枚举空白符，而非直接使用 `\s`

是因为**Flex 的默认正则引擎不完全支持 `\s`**：

- Flex 基于传统的 POSIX 正则，原生不支持 `\s`、`\d`等快捷符号
- 若强行使用 `\s`，需开启扩展模式（如 `%option reentrant`或 `%option bison-bridge`），但会增加复杂性



词法特性相比 C 语言做了大量简化，比如标识符 `student_id` 在 C 语言中是合法的，但是在 Cminusf 中是不合法的

```c
%%
 /* to do for students */
 /* two cases for you, pass_node will send flex's token to bison */
\+ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return ADD;}
. { pos_start = pos_end; pos_end++; return ERROR; }

 /****请在此补全所有flex的模式与动作  end******/
 // 基础运算
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

 // 符号
\; {pos_start = pos_end; pos_end += 1; pass_node(yytext); return SEMICOLON;}
\, {pos_start = pos_end; pos_end += 1; pass_node(yytext); return COMMA;}
\( {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LPARENTHESE;}
\) {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RPARENTHESE;}
\[ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACKET;}
\] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACKET;}
\{ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACE;}
\} {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACE;}

 // 关键字
else {pos_start = pos_end; pos_end += 4; pass_node(yytext); return ELSE;}
if {pos_start = pos_end; pos_end += 2; pass_node(yytext); return IF;}
int {pos_start = pos_end; pos_end += 3; pass_node(yytext); return INT;}
return {pos_start = pos_end; pos_end += 6; pass_node(yytext); return RETURN;}
void {pos_start = pos_end; pos_end += 4; pass_node(yytext); return VOID;}
while {pos_start = pos_end; pos_end += 5; pass_node(yytext); return WHILE;}
float {pos_start = pos_end; pos_end += 5; pass_node(yytext); return FLOAT;}

 // ID & NUM
[a-zA-Z] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LETTER;}
[0-9] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return DIGIT;}
[a-zA-Z]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return ID;}
[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return INTEGER;}
[0-9]+\.|[0-9]*\.[0-9]+ {pos_start = pos_end; pos_end += strlen(yytext); pass_node(yytext); return FLOATPOINT;}

 // 注释等其他特殊符号
\n  {lines++ ; pos_start = 1; pos_end = 1;}
\[\] {pos_start = pos_end; pos_end += 2; pass_node(yytext);return ARRAY;}
\/\*([^\*]|\*+[^*/])*\*+\/ {
     for(int i=0;i<strlen(yytext);i++)
     {
		if(yytext[i]=='\n')
          {
			pos_start=1;
			pos_end=1;
			lines++;
		}
		else pos_end++;
     }
}
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

**问题3：优先选择最长的可能匹配**

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1$ cat ./tests/testcases_general/2-decl_int.cminus 
void main(void) {
    int a;
    return;
}
manbin@compile:~/2023_warm_up_b/_lab1/lab1$ parser ./tests/testcases_general/2-decl_int.cminus 
error at line 2 column 9: syntax error
```

语法分析器刚处理完 `type-specifier ID`

* 如果是函数参数，它需要等待 `,`、`)`或 `ARRAY`

  ```c
  param-list : param-list COMMA param   { $$ = node(...); }  // 逗号分隔多个参数
             | param                   { $$ = node(...); }   // 单个参数
  
  params : param-list { $$ = node("params", 1, $1); }
         | VOID      { $$ = node("params", 1, $1); }  // 无参数情况
  
  fun-declaration : type-specifier ID LPARENTHESE params RPARENTHESE compound-stmt
  { 
      $$
   = node("fun-declaration", 6, $1, $2, $3, $4, $5, $6);
  }
  ```

- 如果是变量声明，它需要等待 `;`或 `[`

  ```c
  param : type-specifier ID { $$ = node("param", 2, $1, $2); }
        | type-specifier ID ARRAY { $$ = node("param", 3, $1, $2, $3); }
  ```

由于**优先选择最长的可能匹配**：分析器更愿意接受`ARRAY`，而不是立即结束

无法确定是规则A（普通参数）还是规则B（数组参数）,它会等待下一个 token 来决定选择哪条规则

它期望看到 `ARRAY`(如果是函数参数数组声明)，但实际输入是 `;`(变量声明结束符)

**为什么SEMICOLON匹配失败？**

解析器错误地将函数体中的变量声明误判为函数参数的延续

> yacc -v syntax_analyzer.y

查看所有语法冲突和状态转换：

```shell
manbin@compile:~/2023_warm_up_b/_lab1/lab1/src/parser$ yacc -v syntax_analyzer.y
syntax_analyzer.y:44.14-27: warning: POSIX Yacc forbids dashes in symbol names: type-specifier [-Wyacc]
   44 | %type <node> type-specifier relop addop mulop
      |              ^~~~~~~~~~~~~~
syntax_analyzer.y:45.14-29: warning: POSIX Yacc forbids dashes in symbol names: declaration-list [-Wyacc]
   45 | %type <node> declaration-list declaration var-declaration fun-declaration local-declarations
      |              ^~~~~~~~~~~~~~~~
syntax_analyzer.y:45.43-57: warning: POSIX Yacc forbids dashes in symbol names: var-declaration [-Wyacc]
   45 | %type <node> declaration-list declaration var-declaration fun-declaration local-declarations
      |                                           ^~~~~~~~~~~~~~~
syntax_analyzer.y:45.59-73: warning: POSIX Yacc forbids dashes in symbol names: fun-declaration [-Wyacc]
   45 | %type <node> declaration-list declaration var-declaration fun-declaration local-declarations
      |                                                           ^~~~~~~~~~~~~~~
syntax_analyzer.y:45.75-92: warning: POSIX Yacc forbids dashes in symbol names: local-declarations [-Wyacc]
   45 | %type <node> declaration-list declaration var-declaration fun-declaration local-declarations
      |                                                                           ^~~~~~~~~~~~~~~~~~
syntax_analyzer.y:46.14-26: warning: POSIX Yacc forbids dashes in symbol names: compound-stmt [-Wyacc]
   46 | %type <node> compound-stmt statement-list statement expression-stmt iteration-stmt selection-stmt return-stmt
      |              ^~~~~~~~~~~~~
syntax_analyzer.y:46.28-41: warning: POSIX Yacc forbids dashes in symbol names: statement-list [-Wyacc]
   46 | %type <node> compound-stmt statement-list statement expression-stmt iteration-stmt selection-stmt return-stmt
      |                            ^~~~~~~~~~~~~~
syntax_analyzer.y:46.53-67: warning: POSIX Yacc forbids dashes in symbol names: expression-stmt [-Wyacc]
   46 | %type <node> compound-stmt statement-list statement expression-stmt iteration-stmt selection-stmt return-stmt
      |                                                     ^~~~~~~~~~~~~~~
syntax_analyzer.y:46.69-82: warning: POSIX Yacc forbids dashes in symbol names: iteration-stmt [-Wyacc]
   46 | %type <node> compound-stmt statement-list statement expression-stmt iteration-stmt selection-stmt return-stmt
      |                                                                     ^~~~~~~~~~~~~~
syntax_analyzer.y:46.84-97: warning: POSIX Yacc forbids dashes in symbol names: selection-stmt [-Wyacc]
   46 | %type <node> compound-stmt statement-list statement expression-stmt iteration-stmt selection-stmt return-stmt
      |                                                                                    ^~~~~~~~~~~~~~
syntax_analyzer.y:46.99-109: warning: POSIX Yacc forbids dashes in symbol names: return-stmt [-Wyacc]
   46 | %type <node> compound-stmt statement-list statement expression-stmt iteration-stmt selection-stmt return-stmt
      |                                                                                                   ^~~~~~~~~~~
syntax_analyzer.y:47.14-30: warning: POSIX Yacc forbids dashes in symbol names: simple-expression [-Wyacc]
   47 | %type <node> simple-expression expression var additive-expression term factor integer float call
      |              ^~~~~~~~~~~~~~~~~
syntax_analyzer.y:47.47-65: warning: POSIX Yacc forbids dashes in symbol names: additive-expression [-Wyacc]
   47 | %type <node> simple-expression expression var additive-expression term factor integer float call
      |                                               ^~~~~~~~~~~~~~~~~~~
syntax_analyzer.y:48.21-30: warning: POSIX Yacc forbids dashes in symbol names: param-list [-Wyacc]
   48 | %type <node> params param-list param args arg-list
      |                     ^~~~~~~~~~
syntax_analyzer.y:48.43-50: warning: POSIX Yacc forbids dashes in symbol names: arg-list [-Wyacc]
   48 | %type <node> params param-list param args arg-list
      |                                           ^~~~~~~~
syntax_analyzer.y: warning: 1 shift/reduce conflict [-Wconflicts-sr]
syntax_analyzer.y: note: rerun with option '-Wcounterexamples' to generate conflict counterexamples
```

可以发现有关于移进/归约冲突的冲突，这也是问题根本原因。其他警告不用管，按照文档来写即可

**增加优先级声明**

```c
%nonassoc LBRACKET
%nonassoc ARRAY
%right SEMICOLON
```

1. 看到 `ARRAY`→ 移进（因为 `ARRAY`优先级高于 `SEMICOLON`）
2. 看到 `SEMICOLON`→ 归约为数组声明



### 2.4 思路

先看实验细节与要求，根据cminusf的文法，列出所有token和type并且根据这个顺序可以去写相关.y文件的规则

因为`%token`定义的符号需要与 Flex 词法分析器的返回值匹配，是词法分析器的接口；

​	`%type`声明的非终结符必须有对应的语法规则实现，用来将Token 组合成非终结符节点，这可以看语法分析的实验内容，一一对应补全

所以token需要在.l文件中一一匹配相应规则，此时再看词法分析的实验内容，将其匹配规则和行为补全
