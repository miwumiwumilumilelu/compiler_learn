#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "def.h"
#include "parser.tab.h" // 包含节点类型定义

// --- 符号表实现 ---
Symbol symbolTable[MAX_SYMBOLS];
int symbolCount = 0;

void initSymbolTable() {
    symbolCount = 0;
}

Symbol* findSymbol(char *name) {
    for (int i = 0; i < symbolCount; i++) {
        if (strcmp(symbolTable[i].name, name) == 0) {
            return &symbolTable[i];
        }
    }
    return NULL;
}

void setSymbolValue(char *name, EvalResult result) {
    Symbol *symbol = findSymbol(name);
    if (symbol == NULL) {
        if (symbolCount >= MAX_SYMBOLS) {
            yyerror("Symbol table overflow");
            exit(1);
        }
        symbol = &symbolTable[symbolCount++];
        symbol->name = strdup(name);
    }
    symbol->type = result.type;
    symbol->value = result.value;
}

EvalResult getSymbolValue(ASTNode *idNode) {
    char *name = idNode->stringValue;
    Symbol *symbol = findSymbol(name);
    EvalResult result;
    
    if (symbol == NULL) {
        // 变量未定义，报告错误并退出
        yyerror("第%d行的符号%s未定义值", idNode->lineno, name);
        exit(1); // 终止程序
    } else {
        // 变量已定义，正常返回值
        result.type = symbol->type;
        result.value = symbol->value;
    }
    return result;
}


// --- 表达式求值 (evalExpression) ---
EvalResult evalExpression(ASTNode *node) {
    EvalResult result;
    if (node == NULL) {
        result.type = TYPE_INT;
        result.value.intValue = 0;
        return result;
    }

    switch (node->nodeType) {
        case NODE_INT:
            result.type = TYPE_INT;
            result.value.intValue = node->intValue;
            break;
        case NODE_FLOAT:
            result.type = TYPE_FLOAT;
            result.value.floatValue = node->floatValue;
            break;
        
        case NODE_ID:
            result = getSymbolValue(node); // 传递整个节点
            break;

        case NODE_ASSIGN: {
            char *varName = node->child->stringValue;
            EvalResult right = evalExpression(node->child->next);
            setSymbolValue(varName, right);
            result = right; 
            break;
        }

        // 算术运算
        case NODE_PLUS:
        case NODE_MINUS:
        case NODE_STAR:
        case NODE_DIV: {
            EvalResult left = evalExpression(node->child);
            EvalResult right = evalExpression(node->child->next);
            
            if (left.type == TYPE_FLOAT || right.type == TYPE_FLOAT) {
                float f_left = (left.type == TYPE_INT) ? (float)left.value.intValue : left.value.floatValue;
                float f_right = (right.type == TYPE_INT) ? (float)right.value.intValue : right.value.floatValue;
                result.type = TYPE_FLOAT;
                
                if (node->nodeType == NODE_PLUS) result.value.floatValue = f_left + f_right;
                else if (node->nodeType == NODE_MINUS) result.value.floatValue = f_left - f_right;
                else if (node->nodeType == NODE_STAR) result.value.floatValue = f_left * f_right;
                else if (node->nodeType == NODE_DIV) result.value.floatValue = f_left / f_right;
            
            } else { 
                result.type = TYPE_INT;
                if (node->nodeType == NODE_PLUS) result.value.intValue = left.value.intValue + right.value.intValue;
                else if (node->nodeType == NODE_MINUS) result.value.intValue = left.value.intValue - right.value.intValue;
                else if (node->nodeType == NODE_STAR) result.value.intValue = left.value.intValue * right.value.intValue;
                else if (node->nodeType == NODE_DIV) result.value.intValue = left.value.intValue / right.value.intValue;
            }
            break;
        }

        // 关系运算
        case NODE_EQ: case NODE_NE: case NODE_GT:
        case NODE_GE: case NODE_LT: case NODE_LE: {
            EvalResult left = evalExpression(node->child);
            EvalResult right = evalExpression(node->child->next);
            result.type = TYPE_INT; 
            
            float f_left = (left.type == TYPE_INT) ? (float)left.value.intValue : left.value.floatValue;
            float f_right = (right.type == TYPE_INT) ? (float)right.value.intValue : right.value.floatValue;

            if (node->nodeType == NODE_EQ) result.value.intValue = (f_left == f_right);
            else if (node->nodeType == NODE_NE) result.value.intValue = (f_left != f_right);
            else if (node->nodeType == NODE_GT) result.value.intValue = (f_left > f_right);
            else if (node->nodeType == NODE_GE) result.value.intValue = (f_left >= f_right);
            else if (node->nodeType == NODE_LT) result.value.intValue = (f_left < f_right);
            else if (node->nodeType == NODE_LE) result.value.intValue = (f_left <= f_right);
            break;
        }
        
        // 单目负号
        case NODE_UMINUS: {
            EvalResult val = evalExpression(node->child);
            if (val.type == TYPE_INT) {
                val.value.intValue = -val.value.intValue;
            } else {
                val.value.floatValue = -val.value.floatValue;
            }
            result = val;
            break;
        }
        
        default:
            result.type = TYPE_INT;
            result.value.intValue = 0;
            break;
    }
    return result;
}


// --- 语句执行 (executeProgram) ---
void executeProgram(ASTNode *node) {
    if (node == NULL) return;

    switch (node->nodeType) {
        case NODE_STMT_LIST: {
            ASTNode *stmt = node->child;
            while (stmt != NULL) {
                executeProgram(stmt);
                stmt = stmt->next;
            }
            break;
        }
        
        case NODE_EXP_STMT:
            evalExpression(node->child);
            break;

        case NODE_PRINT_STMT: {
            EvalResult result = evalExpression(node->child);
            if (result.type == TYPE_INT) {
                printf("%d\n", result.value.intValue);
            } else {
                printf("%f\n", result.value.floatValue);
            }
            break;
        }

        /* --- 关键修改：重写 SCAN 逻辑 --- */
        case NODE_SCAN_STMT: {
            char *varName = node->child->stringValue;
            EvalResult res;
            char buffer[100];
            
            if (scanf("%s", buffer) == 1) { // 1. 先读取为字符串
                // 2. 检查字符串是否包含 '.' 'e' 或 'E'
                if (strchr(buffer, '.') != NULL || strchr(buffer, 'e') != NULL || strchr(buffer, 'E') != NULL) {
                    // 3a. 如果是，则当作浮点数处理
                    res.type = TYPE_FLOAT;
                    res.value.floatValue = atof(buffer);
                } else {
                    // 3b. 否则，当作整数处理
                    res.type = TYPE_INT;
                    res.value.intValue = atoi(buffer);
                }
                setSymbolValue(varName, res);
            }
            break;
        }

        // IF 语句
        case NODE_IF_STMT: {
            EvalResult cond = evalExpression(node->child);
            int isTrue = (cond.type == TYPE_INT) ? cond.value.intValue != 0 : cond.value.floatValue != 0.0;
            if (isTrue) {
                executeProgram(node->child->next);
            }
            break;
        }

        // IF-ELSE 语句
        case NODE_IF_ELSE_STMT: {
            EvalResult cond = evalExpression(node->child);
            int isTrue = (cond.type == TYPE_INT) ? cond.value.intValue != 0 : cond.value.floatValue != 0.0;
            if (isTrue) {
                executeProgram(node->child->next);
            } else {
                executeProgram(node->child->next->next);
            }
            break;
        }

        // WHILE 语句
        case NODE_WHILE_STMT: {
            while (1) {
                EvalResult cond = evalExpression(node->child);
                int isTrue = (cond.type == TYPE_INT) ? cond.value.intValue != 0 : cond.value.floatValue != 0.0;
                if (!isTrue) {
                    break;
                }
                executeProgram(node->child->next);
            }
            break;
        }
    }
}