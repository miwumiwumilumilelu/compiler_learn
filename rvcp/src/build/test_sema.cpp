#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <map>
#include <cassert>

// 引入所有前端组件头文件
#include "../parse/Lexer.h"
#include "../parse/Parser.h"
#include "../parse/Sema.h"         // 语义分析
#include "../parse/TypeContext.h"
#include "../parse/ASTNode.h"      // 抽象语法树
#include "../utils/DynamicCast.h" 

using namespace sys;

//
// -----------------------------------------------------------------
// 模块 1：Sema-Aware ASTPrinter (用于验证类型和结构)
// -----------------------------------------------------------------
//
class ASTPrinter {
    int indent = 0;
    
    // 打印缩进
    void printIndent() { std::cout << std::string(indent * 2, ' '); }

    // 获取节点类型字符串
    std::string getTypeString(ASTNode *node) {
        return node->type ? node->type->toString() : "NoType";
    }
    
    // 打印节点头，包含推导出的类型
    void printHeader(const std::string &name, ASTNode *node) {
        printIndent();
        std::cout << "- " << name << " (Type: " << getTypeString(node) << ")\n";
    }

    // 递归访问子节点（增加缩进）
    void visitChild(ASTNode *child, const std::string& label = "") {
        if (!label.empty()) {
            printIndent(); 
            std::cout << "  (" << label << "):\n";
        }
        indent++;
        visit(child);
        indent--;
    }

public:
    // 主分发函数
    void visit(ASTNode *node) {
        if (!node) {
            printIndent();
            std::cout << "<nullptr>\n";
            return;
        }

        // --- 核心 AST 节点检查，重点打印 Sema 结果 ---
        
        if (auto n = dyn_cast<IntNode>(node)) {
            printHeader("IntNode (Value: " + std::to_string(n->value) + ")", node);
        } else if (auto n = dyn_cast<FloatNode>(node)) {
            printHeader("FloatNode (Value: " + std::to_string(n->value) + ")", node);
        } else if (auto n = dyn_cast<VarRefNode>(node)) {
            printHeader("VarRefNode", node);
            printIndent(); std::cout << "    Name: " << n->name << "\n";
        } else if (auto n = dyn_cast<VarDeclNode>(node)) {
            printHeader("VarDeclNode", node);
            printIndent(); std::cout << "    Name: " << n->name << ", Mut: " << n->mut << ", Global: " << n->global << "\n";
            visitChild(n->init, "Init Value");
        } else if (auto n = dyn_cast<BinaryNode>(node)) {
            printHeader("BinaryNode", node);
            printIndent(); std::cout << "    Kind: " << n->kind << " (Add=0, Ne=9, And=5, etc.)\n";
            visitChild(n->l, "LHS");
            visitChild(n->r, "RHS");
        } else if (auto n = dyn_cast<UnaryNode>(node)) {
            printHeader("UnaryNode", node);
            printIndent(); std::cout << "    Kind: " << n->kind << " (F2I=2, I2F=3)\n";
            visitChild(n->node, "Operand");
        } else if (auto n = dyn_cast<FnDeclNode>(node)) {
            printHeader("FnDeclNode", node);
            printIndent(); std::cout << "    Name: " << n->name << ", Params: " << n->params.size() << "\n";
            visitChild(n->body, "Body");
        } else if (auto n = dyn_cast<ReturnNode>(node)) {
            printHeader("ReturnNode", node);
            visitChild(n->node, "Value");
        } else if (auto n = dyn_cast<IfNode>(node)) {
            printHeader("IfNode", node);
            visitChild(n->cond, "Condition");
            visitChild(n->ifso, "If-So Block");
            visitChild(n->ifnot, "Else Block");
        } else if (auto n = dyn_cast<WhileNode>(node)) {
            printHeader("WhileNode", node);
            visitChild(n->cond, "Condition");
            visitChild(n->body, "Body");
        } else if (auto n = dyn_cast<BlockNode>(node)) {
            printHeader("BlockNode (Scoped)", node);
            for (auto sub_node : n->nodes) {
                visitChild(sub_node);
            }
        // FIX: 关键的顶层根节点处理 (如果 Parser 返回这个类型)
        } else if (auto n = dyn_cast<TransparentBlockNode>(node)) {
            printHeader("TransparentBlockNode (ROOT)", node);
            // TransparentBlockNode 包含的是 VarDeclNode*，需要正确迭代
            for (auto sub_node : n->nodes) {
                visitChild(sub_node);
            }
        } else if (auto n = dyn_cast<ConstArrayNode>(node)) {
            printHeader("ConstArrayNode", node);
            printIndent(); std::cout << "    IsFloat: " << n->isFloat << "\n";
            // 静态常量数组，值通常不在此打印
        } else if (auto n = dyn_cast<LocalArrayNode>(node)) {
            printHeader("LocalArrayNode", node);
            // ... (保持现有 LocalArrayNode 逻辑)
        } else if (auto n = dyn_cast<ArrayAccessNode>(node)) {
            // ID: 170
            printHeader("ArrayAccessNode", node);
            printIndent(); std::cout << "    Array: " << n->array << "\n";
            printIndent(); std::cout << "    ArrayType (Sema): " << (n->arrayType ? n->arrayType->toString() : "NoTy") << "\n";
            printIndent(); std::cout << "    (Indices):\n";
            for (auto idx : n->indices) {
                visitChild(idx);
            }
        } else if (auto n = dyn_cast<ArrayAssignNode>(node)) {
            // ID: 182
            printHeader("ArrayAssignNode", node);
            printIndent(); std::cout << "    Array: " << n->array << "\n";
            printIndent(); std::cout << "    (Indices):\n";
            for (auto idx : n->indices) {
                visitChild(idx);
            }
            visitChild(n->value, "RHS Value");
        } else if (auto n = dyn_cast<WhileNode>(node)) {
            printHeader("WhileNode", node);
            visitChild(n->cond, "Condition");
            visitChild(n->body, "Body");
        } else if (auto n = dyn_cast<AssignNode>(node)) {
            // ID 138: AssignNode 逻辑
            printHeader("AssignNode", node);
            visitChild(n->l, "Left Side");
            visitChild(n->r, "Right Side");
        } else if (auto n = dyn_cast<CallNode>(node)) {
            // ID 196: CallNode 逻辑
            printHeader("CallNode", node);
            printIndent(); std::cout << "    Name: " << n->callee << "\n";
            printIndent(); std::cout << "    (Args):\n";
            for (auto arg : n->args) {
                visitChild(arg);
            }
        }
        else {
            printHeader("!!! UNKNOWN ASTNode (ID: " + std::to_string(node->getID()) + ") !!!", node);
        }
    }
};

//
// -----------------------------------------------------------------
// 模块 2：测试驱动程序 (main)
// -----------------------------------------------------------------
//
int run_test_from_file(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }

    // 将整个文件读入字符串
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source_code = buffer.str();

    std::cout << "=== Starting Sema Analysis for: " << filename << " ===\n\n";

    try {
        // 1. 创建类型工厂
        TypeContext ctx;
        
        // 2. Lexer & Parser 
        // 假设 Parser(source_code, ctx) 包含了 Lexer 的初始化
        Parser parser(source_code, ctx);
        ASTNode *root = parser.parse();

        if (!root) {
            std::cerr << "Error: Failed to parse file " << filename << "\n";
            return 1;
        }

        // 3. 运行 Sema (核心测试步骤)
        Sema sema(root, ctx);

        std::cout << "=== Sema Analysis Complete. Resulting AST: ===\n\n";

        // 4. 打印 AST，验证 Sema 结果
        ASTPrinter printer;
        printer.visit(root);

        // 5. 清理内存 
        delete root; 
        
        std::cout << "\n=== Test Finished Successfully. ===\n" << std::endl;
        
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error during compilation pass: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file.sy>\n";
        return 1;
    }
    
    return run_test_from_file(argv[1]);
}