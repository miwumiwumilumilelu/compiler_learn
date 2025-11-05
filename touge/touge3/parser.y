%error-verbose
%locations
%{
#include "stdio.h"
#include "math.h"
#include "string.h"
#include "def.h"
extern int yylineno;
extern char *yytext;
extern FILE *yyin;
void yyerror(const char* fmt, ...);
int yylex();

struct ASTNode *root = NULL;

struct ASTNode *createNode(int nodeType, int line) {
    struct ASTNode *node = (struct ASTNode *)malloc(sizeof(struct ASTNode));
    if (!node) { yyerror("out of memory"); exit(0); }
    node->nodeType = nodeType;
    node->intValue = 0;
    node->floatValue = 0.0;
    node->stringValue = NULL;
    node->child = NULL;
    node->next = NULL;
    node->lineno = line; 
    return node;
}
struct ASTNode *createIntNode(int value, int line) {
    struct ASTNode *node = createNode(NODE_INT, line);
    node->intValue = value;
    return node;
}
struct ASTNode *createFloatNode(float value, int line) {
    struct ASTNode *node = createNode(NODE_FLOAT, line);
    node->floatValue = value;
    return node;
}
struct ASTNode *createIdNode(char *value, int line) {
    struct ASTNode *node = createNode(NODE_ID, line);
    node->stringValue = strdup(value);
    return node;
}
void addChild(struct ASTNode *parent, struct ASTNode *child) {
    if (parent->child == NULL) {
        parent->child = child;
    } else {
        struct ASTNode *temp = parent->child;
        while (temp->next != NULL) { temp = temp->next; }
        temp->next = child;
    }
}
%}

%union {
    int type_int;
    float type_float;
    char type_id[32];
    struct ASTNode *node;
}
%token <type_int> INT
%token <type_float> FLOAT
%token <type_id> ID
%token IF ELSE WHILE SCAN PRINT
%token LC RC LP RP SEMI
%token ASSIGNOP PLUS MINUS STAR DIV
%token EQ NE GT GE LT LE

%type <node> program StmList Stmt CompSt Exp
%nonassoc LOWER_THAN_ELSE
%nonassoc ELSE
%left ASSIGNOP
%left EQ NE GT GE LT LE
%left PLUS MINUS
%left STAR DIV
%right UMINUS
%start program

%%

program: StmList { root = $1; }
    ;
StmList: /* empty */ { $$ = createNode(NODE_STMT_LIST, yylloc.first_line); }
    | StmList Stmt { addChild($1, $2); $$ = $1; }
    ;
CompSt: LC StmList RC { $$ = $2; }
    ;
Stmt: Exp SEMI { 
        $$ = createNode(NODE_EXP_STMT, @1.first_line);
        addChild($$, $1);
      }
    | CompSt { $$ = $1; }
    | SCAN ID SEMI { 
        $$ = createNode(NODE_SCAN_STMT, @1.first_line);
        addChild($$, createIdNode($2, @2.first_line));
      }
    | PRINT Exp SEMI { 
        $$ = createNode(NODE_PRINT_STMT, @1.first_line);
        addChild($$, $2);
      }
    | IF LP Exp RP Stmt %prec LOWER_THAN_ELSE { 
        $$ = createNode(NODE_IF_STMT, @1.first_line);
        addChild($$, $3);
        addChild($$, $5);
      }
    | IF LP Exp RP Stmt ELSE Stmt { 
        $$ = createNode(NODE_IF_ELSE_STMT, @1.first_line);
        addChild($$, $3);
        addChild($$, $5);
        addChild($$, $7);
      }
    | WHILE LP Exp RP Stmt { 
        $$ = createNode(NODE_WHILE_STMT, @1.first_line);
        addChild($$, $3);
        addChild($$, $5);
      }
    ;
Exp: INT { $$ = createIntNode($1, @1.first_line); }
    | FLOAT { $$ = createFloatNode($1, @1.first_line); }
    | ID { $$ = createIdNode($1, @1.first_line); }
    
    | Exp ASSIGNOP Exp { 
        $$ = createNode(NODE_ASSIGN, @2.first_line); 
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp PLUS Exp { 
        $$ = createNode(NODE_PLUS, @2.first_line); 
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp MINUS Exp { 
        $$ = createNode(NODE_MINUS, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp STAR Exp { 
        $$ = createNode(NODE_STAR, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp DIV Exp { 
        $$ = createNode(NODE_DIV, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp EQ Exp { 
        $$ = createNode(NODE_EQ, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp NE Exp { 
        $$ = createNode(NODE_NE, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp GT Exp { 
        $$ = createNode(NODE_GT, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp GE Exp { 
        $$ = createNode(NODE_GE, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp LT Exp { 
        $$ = createNode(NODE_LT, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | Exp LE Exp { 
        $$ = createNode(NODE_LE, @2.first_line);
        addChild($$, $1); addChild($$, $3); 
      }
    | MINUS Exp %prec UMINUS { 
        $$ = createNode(NODE_UMINUS, @1.first_line);
        addChild($$, $2);
      }
    | LP Exp RP { $$ = $2; }
    ;

%%

int main(int argc, char *argv[]){
    if (argc <= 1) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }
    yyin = fopen(argv[1], "r");
    if (!yyin) {
        perror(argv[1]);
        return 1;
    }
    yylineno = 1;
    yyparse(); 
    
    initSymbolTable();
    executeProgram(root);

    fclose(yyin);
    return 0;
}

#include<stdarg.h>
/* --- 关键修改：重写 yyerror 逻辑 --- */
void yyerror(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);

    // 检查 fmt 字符串是否是来自 semantics.c 的自定义错误
    // "第" 在 UTF-8 中是 3 字节: 0xE7 0xAC 0xAC
    if (strncmp(fmt, "\xE7\xAC\xAC", 3) == 0) {
        // 如果是自定义错误 (如 "第...行..."), 则直接按原样打印到 stderr
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n"); // 确保换行
    } else {
        // 否则, 认为是 Bison 传来的“语法错误”
        // 打印完整的 "Grammar Error at..." 前缀
        fprintf(stderr, "Grammar Error at Line %d Column %d: ", yylloc.first_line, yylloc.first_column);
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, " near '%s'\n", yytext);
    }
    
    va_end(ap);
}