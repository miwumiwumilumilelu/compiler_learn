#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// 包含所有你需要的文件
#include "../parse/Parser.h"
#include "../parse/Lexer.h"
#include "../parse/TypeContext.h"
#include "../parse/ASTNode.h"
#include "../utils/DynamicCast.h"

//
// -----------------------------------------------------------------
// 模块 1：ASTPrinter 辅助类
// -----------------------------------------------------------------
//
class ASTPrinter {
    int indent = 0;
    
    // 打印缩进
    void printIndent() { std::cout << std::string(indent * 2, ' '); }

    // 打印节点类型和基本信息
    void printHeader(const std::string &name) {
        printIndent();
        std::cout << name << "\n";
    }

    // 递归访问子节点（增加缩进）
    void visitChild(sys::ASTNode *child) {
        indent++;
        visit(child);
        indent--;
    }

public:
    // 主分发函数 (使用 dyn_cast 的手动递归)
    void visit(sys::ASTNode *node) {
        using namespace sys; // 允许我们直接使用 IntNode, BinaryNode 等

        if (!node) {
            printIndent();
            std::cout << "<nullptr>\n";
            return;
        }

        // --- 检查每一种 ASTNode 类型 ---
        
        // 叶节点 (Leaf Nodes)
        if (auto n = dyn_cast<IntNode>(node)) {
            printIndent();
            std::cout << "IntNode (value: " << n->value << ")\n";
        } else if (auto n = dyn_cast<FloatNode>(node)) {
            printIndent();
            std::cout << "FloatNode (value: " << n->value << ")\n";
        } else if (auto n = dyn_cast<VarRefNode>(node)) {
            printIndent();
            std::cout << "VarRefNode (name: " << n->name << ")\n";
        } else if (isa<BreakNode>(node)) {
            printHeader("BreakNode");
        } else if (isa<ContinueNode>(node)) {
            printHeader("ContinueNode");
        } else if (isa<EmptyNode>(node)) {
            printHeader("EmptyNode");
        
        // 分支节点 (Branch Nodes)
        } else if (auto n = dyn_cast<BlockNode>(node)) {
            printHeader("BlockNode (scoped)");
            for (auto sub_node : n->nodes) {
                visitChild(sub_node);
            }
        } else if (auto n = dyn_cast<TransparentBlockNode>(node)) {
            printHeader("TransparentBlockNode (no scope)");
            for (auto sub_node : n->nodes) {
                visitChild(sub_node);
            }
        } else if (auto n = dyn_cast<VarDeclNode>(node)) {
            printIndent();
            std::cout << "VarDeclNode (name: " << n->name << ", mut: " << n->mut << ", global: " << n->global << ")\n";
            printIndent(); std::cout << "  (type: " << n->type->toString() << ")\n";
            printIndent(); std::cout << "  (init):\n";
            visitChild(n->init);
        } else if (auto n = dyn_cast<BinaryNode>(node)) {
            printIndent();
            std::cout << "BinaryNode (kind: " << n->kind << ")\n"; // kind 是 enum，会打印数字
            visitChild(n->l);
            visitChild(n->r);
        } else if (auto n = dyn_cast<UnaryNode>(node)) {
            printIndent();
            std::cout << "UnaryNode (kind: " << n->kind << ")\n";
            visitChild(n->node);
        } else if (auto n = dyn_cast<FnDeclNode>(node)) {
            printIndent();
            std::cout << "FnDeclNode (name: " << n->name << ")\n";
            printIndent(); std::cout << "  (type: " << n->type->toString() << ")\n";
            printIndent(); std::cout << "  (body):\n";
            visitChild(n->body);
        } else if (auto n = dyn_cast<ReturnNode>(node)) {
            printHeader("ReturnNode");
            visitChild(n->node);
        } else if (auto n = dyn_cast<IfNode>(node)) {
            printHeader("IfNode");
            printIndent(); std::cout << "  (cond):\n";
            visitChild(n->cond);
            printIndent(); std::cout << "  (ifso):\n";
            visitChild(n->ifso);
            printIndent(); std::cout << "  (ifnot):\n";
            visitChild(n->ifnot);
        } else if (auto n = dyn_cast<WhileNode>(node)) {
            printHeader("WhileNode");
            printIndent(); std::cout << "  (cond):\n";
            visitChild(n->cond);
            printIndent(); std::cout << "  (body):\n";
            visitChild(n->body);
        } else if (auto n = dyn_cast<AssignNode>(node)) {
            printHeader("AssignNode");
            printIndent(); std::cout << "  (left):\n";
            visitChild(n->l);
            printIndent(); std::cout << "  (right):\n";
            visitChild(n->r);
        } else if (auto n = dyn_cast<CallNode>(node)) {
            printIndent();
            std::cout << "CallNode (name: " << n->callee << ")\n";
            printIndent(); std::cout << "  (args):\n";
            for (auto arg : n->args) {
                visitChild(arg);
            }
        } else if (auto n = dyn_cast<ArrayAccessNode>(node)) {
            printIndent();
            std::cout << "ArrayAccessNode (name: " << n->array << ")\n";
            printIndent(); std::cout << "  (indices):\n";
            for (auto idx : n->indices) {
                visitChild(idx);
            }
        } else if (auto n = dyn_cast<ArrayAssignNode>(node)) {
            printIndent();
            std::cout << "ArrayAssignNode (name: " << n->array << ")\n";
            printIndent(); std::cout << "  (indices):\n";
            for (auto idx : n->indices) {
                visitChild(idx);
            }
            printIndent(); std::cout << "  (value):\n";
            visitChild(n->value);
        } else if (auto n = dyn_cast<ConstArrayNode>(node)) {
            printIndent();
            std::cout << "ConstArrayNode (isFloat: " << n->isFloat << ", type: " 
                      << (n->type ? n->type->toString() : "nullptr") << ")\n";
            // 注意：ConstArrayNode 内存由其管理，这里只打印类型信息
        } else if (auto n = dyn_cast<LocalArrayNode>(node)) {
            printIndent();
            std::cout << "LocalArrayNode (type: " 
                      << (n->type ? n->type->toString() : "nullptr") << ")\n";
            printIndent(); std::cout << "  (elements):\n";
            // 遍历元素数组（注意：需要知道元素个数）
            // 这里假设 elements 被正确初始化
            if (n->elements) {
                int i = 0;
                while (n->elements[i] != nullptr) {
                    visitChild(n->elements[i]);
                    i++;
                }
            }
        }
        
        else {
            printIndent();
            std::cout << "!!! UNKNOWN ASTNode (ID: " << node->getID() << ") !!!\n";
        }
    }
};

//
// -----------------------------------------------------------------
// 模块 2：测试驱动程序 (main)
// -----------------------------------------------------------------
//
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename.sy>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../test/custom/timer.manbin" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }

    // 将整个文件一次性读入字符串
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source_code = buffer.str();

    std::cout << "=== Parsing file: " << filename << " ===\n\n";

    try {
        // 1. 创建类型工厂
        sys::TypeContext ctx;
        
        // 2. 创建 Parser 实例
        // (构造函数 会自动调用 Lexer 并填充所有 Token)
        sys::Parser parser(source_code, ctx);
        
        // 3. 运行解析！
        sys::ASTNode *root = parser.parse();

        std::cout << "=== Parse Complete. AST Structure: ===\n\n";

        // 4. 打印 AST
        ASTPrinter printer;
        printer.visit(root);

        std::cout << "\n=== Cleaning up AST... ===\n";
        
        // 5. 清理内存
        // (这将递归调用所有子节点的析构函数)
        delete root; 
        
        std::cout << "=== AST Cleaned Successfully. ===\n" << std::endl;
        
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error during parsing: " << e.what() << std::endl;
        return 1;
    }
    // (当 main 结束时，'ctx' 会被销毁，
    // 其析构函数会释放所有 Type 内存)
}