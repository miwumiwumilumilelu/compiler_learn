#include <iostream>
#include <cassert>
#include <vector>

// 包含你的基础文件
// (我们不需要 Type.cpp 或 TypeContext.h，因为我们只存储 Type*)
#include "../utils/DynamicCast.h"
#include "../parse/Type.h"
#include "../parse/ASTNode.h"

// 使用你自己的命名空间
using namespace sys;

int main() {
    std::cout << "--- Testing ASTNode System ---" << std::endl;

    // --- 测试 1: RTTI (isa/dyn_cast) ---
    std::cout << "Testing RTTI..." << std::endl;
    ASTNode* base = new IntNode(10);
    
    // 检查 isa<>
    assert(isa<IntNode>(base));
    assert(!isa<FloatNode>(base));
    
    // 检查 dyn_cast<> 成功
    IntNode* intNode = dyn_cast<IntNode>(base);
    assert(intNode != nullptr);
    assert(intNode->value == 10);

    // 检查 dyn_cast<> 失败
    FloatNode* floatNode = dyn_cast<FloatNode>(base);
    assert(floatNode == nullptr);

    delete base; // 测试 IntNode 析构函数
    std::cout << "RTTI OK." << std::endl;

    // --- 测试 2: 递归内存管理 (核心测试) ---
    std::cout << "Testing recursive destructors..." << std::endl;
    { // 创建一个新作用域
        
        // 1. 创建一个根
        BlockNode* root = new BlockNode({});

        // 2. 创建一个复杂的子树: if (1+2) { return 3; }
        ASTNode* cond = new BinaryNode(BinaryNode::Add, new IntNode(1), new IntNode(2));
        ASTNode* body = new BlockNode({ new ReturnNode("main", new IntNode(3)) });
        IfNode* ifStmt = new IfNode(cond, body, nullptr);
        
        // 3. 创建另一个子树: int a = 4.0;
        VarDeclNode* varDecl = new VarDeclNode("a", new FloatNode(4.0));
        
        // 4. 将子树添加到根
        root->nodes.push_back(ifStmt);
        root->nodes.push_back(varDecl);
        
        std::cout << "  AST created. Deleting root BlockNode..." << std::endl;
        
        // 5. 【测试点】删除根节点
        delete root; 
        
        // 如果 ASTNode.h 中的析构函数都正确，
        // 这里的 delete root 会递归地删除：
        // - root (BlockNode)
        //   - ifStmt (IfNode)
        //     - cond (BinaryNode)
        //       - IntNode(1)
        //       - IntNode(2)
        //     - body (BlockNode)
        //       - ReturnNode
        //         - IntNode(3)
        //   - varDecl (VarDeclNode)
        //     - FloatNode(4.0)
        
    } // root 在这里被销毁
    std::cout << "Recursive destructors OK (if no crash/leaks)." << std::endl;

    // --- 测试 3: 测试你修复的内存泄漏 ---
    // (这个测试假设你已经为 CallNode, ArrayAccessNode 等添加了析构函数)
    std::cout << "Testing fixed leak nodes (Call, ArrayAccess, ArrayAssign)..." << std::endl;
    {
        // 1. Test CallNode
        std::vector<ASTNode*> args = {new IntNode(10), new VarRefNode("x")};
        CallNode* call = new CallNode("myFunc", args);
        
        // 2. Test ArrayAccess
        std::vector<ASTNode*> indices = {new IntNode(0)};
        ArrayAccessNode* access = new ArrayAccessNode("myArray", indices);
        
        // 3. Test ArrayAssign
        std::vector<ASTNode*> indices2 = {new IntNode(1)};
        ASTNode* value = new IntNode(100);
        ArrayAssignNode* assign = new ArrayAssignNode("myArray", indices2, value);

        // 我们把它们都放进一个 BlockNode 来测试
        BlockNode* leak_test_block = new BlockNode({call, access, assign});
        
        std::cout << "  Deleting block containing leak-prone nodes..." << std::endl;
        delete leak_test_block;
        
        // 如果你修复了 Bug，这个 delete 会级联删除
        // call, access, assign，以及它们所有的子节点
        // (args, indices, indices2, value)
    }
    std::cout << "Leak node test finished." << std::endl;


    std::cout << "--- All ASTNode Tests Finished ---" << std::endl;
    return 0;
}