#include "stdio.h"
#include "def.h"
#include "parser.tab.h" // 包含节点类型定义

// 打印缩进的辅助函数
void print_indent(int indent) {
    for (int i = 0; i < indent; i++) {
        printf("    ");
    }
}

void displayAST(struct ASTNode *node, int indent) {
    if (node == NULL) return;

    // 语句列表特殊处理：直接遍历打印其下的各个语句
    if (node->nodeType == NODE_STMT_LIST) {
        struct ASTNode *child = node->child;
        while (child != NULL) {
            displayAST(child, indent);
            child = child->next;
        }
        return;
    }

    print_indent(indent);

    // 根据结点类型打印
    switch (node->nodeType) {
        case NODE_EXP_STMT:
            printf("表达式语句：\n");
            break;
        case NODE_SCAN_STMT:
            printf("输入变量：%s\n", node->child->stringValue);
            return; // 已处理子节点，直接返回
        case NODE_PRINT_STMT:
            printf("输出表达式:\n");
            break;
        case NODE_IF_STMT:
            printf("条件语句(if_then)：\n");
            print_indent(indent + 1);
            printf("条件：\n");
            displayAST(node->child, indent + 2);
            print_indent(indent + 1);
            printf("if子句：\n");
            displayAST(node->child->next, indent + 2);
            return; // 手动处理子节点，返回
        case NODE_IF_ELSE_STMT:
            printf("条件语句(if_then_else)：\n");
            print_indent(indent + 1);
            printf("条件：\n");
            displayAST(node->child, indent + 2);
            print_indent(indent + 1);
            printf("if子句：\n");
            displayAST(node->child->next, indent + 2);
            print_indent(indent + 1);
            printf("else子句：\n");
            displayAST(node->child->next->next, indent + 2);
            return; // 手动处理子节点，返回
        case NODE_WHILE_STMT:
            printf("循环语句：\n");
            print_indent(indent + 1);
            printf("条件：\n");
            displayAST(node->child, indent + 2);
            print_indent(indent + 1);
            printf("循环体：\n");
            displayAST(node->child->next, indent + 2);
            return; // 手动处理子节点，返回
        
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
        case NODE_UMINUS: printf("UMINUS (单目-)\n"); break;
        
        default:
            printf("未知节点类型: %d\n", node->nodeType);
            break;
    }

    // 递归遍历子结点
    struct ASTNode *child = node->child;
    while (child != NULL) {
        displayAST(child, indent + 1);
        child = child->next;
    }
}