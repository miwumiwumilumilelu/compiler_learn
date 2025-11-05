#ifndef DEF_H
#define DEF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    NODE_PROGRAM,
    NODE_STMT_LIST,
    NODE_COMP_ST,
    NODE_EXP_STMT,
    NODE_SCAN_STMT,
    NODE_PRINT_STMT,
    NODE_IF_STMT,
    NODE_IF_ELSE_STMT,
    NODE_WHILE_STMT,
    NODE_INT,
    NODE_FLOAT,
    NODE_ID,
    NODE_ASSIGN,
    NODE_PLUS,
    NODE_MINUS,
    NODE_STAR,
    NODE_DIV,
    NODE_EQ,
    NODE_NE,
    NODE_GT,
    NODE_GE,
    NODE_LT,
    NODE_LE,
    NODE_UMINUS
};

// 语法树结点类型
typedef struct ASTNode {
    int nodeType;
    int intValue;
    float floatValue;
    char *stringValue;
    struct ASTNode *child;
    struct ASTNode *next;
    int lineno;
} ASTNode;

// 变量类型 (无改动)
enum VarType {
    TYPE_INT,
    TYPE_FLOAT
};

// 统一的值存储联合体 (修复了匿名 union 错误)
typedef union {
    int intValue;
    float floatValue;
} ValueUnion;

// 表达式求值结果的结构体
typedef struct {
    int type;
    ValueUnion value; // <-- 使用命名的 union
} EvalResult;

// 符号表条目的结构体
typedef struct Symbol {
    char *name;
    int type;
    ValueUnion value; // <-- 使用同一个命名的 union
} Symbol;

// --- 符号表声明 ---
#define MAX_SYMBOLS 100
extern Symbol symbolTable[MAX_SYMBOLS];
extern int symbolCount;

void initSymbolTable();
Symbol* findSymbol(char *name);
void setSymbolValue(char *name, EvalResult result);
EvalResult getSymbolValue(ASTNode *idNode);


// --- 语义分析函数 ---
void executeProgram(ASTNode *node);
EvalResult evalExpression(ASTNode *node);

// --- 错误处理函数原型 ---
void yyerror(const char* fmt, ...);

// AST 打印函数原型 (用于调试)
void displayAST(ASTNode *node, int indent);

#endif