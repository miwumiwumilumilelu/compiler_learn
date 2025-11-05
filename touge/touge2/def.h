#ifndef DEF_H
#define DEF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 语法树结点类型
typedef struct ASTNode {
    int nodeType;       // 结点类型
    int intValue;       // 整型值
    float floatValue;   // 浮点值
    char *stringValue;  // 字符串值（用于标识符）
    struct ASTNode *child;   // 第一个子结点
    struct ASTNode *next;    // 下一个兄弟结点
} ASTNode;

// 结点类型枚举
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

// 函数声明
void displayAST(ASTNode *node, int indent);
ASTNode *createNode(int nodeType);
ASTNode *createIntNode(int value);
ASTNode *createFloatNode(float value);
ASTNode *createIdNode(char *value);

#endif