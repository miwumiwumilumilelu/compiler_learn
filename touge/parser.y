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

// 全局根结点
struct ASTNode *root = NULL;

// 创建结点函数
struct ASTNode *createNode(int nodeType) {
    struct ASTNode *node = (struct ASTNode *)malloc(sizeof(struct ASTNode));
    if (!node) {
        yyerror("out of memory");
        exit(0);
    }
    node->nodeType = nodeType;
    node->intValue = 0;
    node->floatValue = 0.0;
    node->stringValue = NULL;
    node->child = NULL;
    node->next = NULL;
    return node;
}

struct ASTNode *createIntNode(int value) {
    struct ASTNode *node = createNode(NODE_INT);
    node->intValue = value;
    return node;
}

struct ASTNode *createFloatNode(float value) {
    struct ASTNode *node = createNode(NODE_FLOAT);
    node->floatValue = value;
    return node;
}

struct ASTNode *createIdNode(char *value) {
    struct ASTNode *node = createNode(NODE_ID);
    node->stringValue = strdup(value);
    return node;
}

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
%token <node> ASSIGNOP PLUS MINUS STAR DIV
%token <node> EQ NE GT GE LT LE

%type <node> program StmList Stmt CompSt Exp

// 定义运算符优先级
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

StmList: { $$ = createNode(NODE_STMT_LIST); }
    | StmList Stmt { addChild($1, $2); $$ = $1; }
    ;

CompSt: LC StmList RC { $$ = $2; }
    ;

Stmt: Exp SEMI { 
        $$ = createNode(NODE_EXP_STMT);
        addChild($$, $1);
      }
    | CompSt { $$ = $1; }
    | SCAN ID SEMI { 
        $$ = createNode(NODE_SCAN_STMT);
        addChild($$, createIdNode($2));
      }
    | PRINT Exp SEMI { 
        $$ = createNode(NODE_PRINT_STMT);
        addChild($$, $2);
      }
    | IF LP Exp RP Stmt %prec LOWER_THAN_ELSE { 
        $$ = createNode(NODE_IF_STMT);
        addChild($$, $3);
        addChild($$, $5);
      }
    | IF LP Exp RP Stmt ELSE Stmt { 
        $$ = createNode(NODE_IF_ELSE_STMT);
        addChild($$, $3);
        addChild($$, $5);
        addChild($$, $7);
      }
    | WHILE LP Exp RP Stmt { 
        $$ = createNode(NODE_WHILE_STMT);
        addChild($$, $3);
        addChild($$, $5);
      }
    ;

Exp: INT { $$ = createIntNode($1); }
    | FLOAT { $$ = createFloatNode($1); }
    | ID { $$ = createIdNode($1); }
    | Exp ASSIGNOP Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp PLUS Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp MINUS Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp STAR Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp DIV Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp EQ Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp NE Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp GT Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp GE Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp LT Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | Exp LE Exp { $$ = $2; addChild($$, $1); addChild($$, $3); }
    | MINUS Exp %prec UMINUS { 
        $$ = createNode(NODE_UMINUS);
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
    displayAST(root, 0);
    fclose(yyin);
    return 0;
}

#include<stdarg.h>
void yyerror(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "Grammar Error at Line %d Column %d: ", yylloc.first_line, yylloc.first_column);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, " near '%s'\n", yytext);
    va_end(ap);
}