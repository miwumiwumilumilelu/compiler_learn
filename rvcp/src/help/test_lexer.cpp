#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "../parse/Lexer.h"

std::string tokenToString(const sys::Token &tok) {
    std::stringstream ss;
    
    // 1. 打印 Token 类型
    switch (tok.type) {
        case sys::Token::LInt:    ss << "LInt";    break;
        case sys::Token::LFloat:  ss << "LFloat";  break;
        case sys::Token::Ident:   ss << "Ident";   break;
        case sys::Token::If:      ss << "If";      break;
        case sys::Token::Else:    ss << "Else";    break;
        case sys::Token::While:   ss << "While";   break;
        case sys::Token::Return:  ss << "Return";  break;
        case sys::Token::For:     ss << "For";     break;
        case sys::Token::Int:     ss << "Int";     break;
        case sys::Token::Float:   ss << "Float";   break;
        case sys::Token::Void:    ss << "Void";    break;
        case sys::Token::Const:   ss << "Const";   break;
        case sys::Token::Break:   ss << "Break";   break;
        case sys::Token::Continue: ss << "Continue"; break;
        case sys::Token::Minus:   ss << "Minus";   break;
        case sys::Token::Plus:    ss << "Plus";    break;
        case sys::Token::Mul:     ss << "Mul";     break;
        case sys::Token::Div:     ss << "Div";     break;
        case sys::Token::Mod:     ss << "Mod";     break;
        case sys::Token::PlusEq:  ss << "PlusEq";  break;
        case sys::Token::MinusEq: ss << "MinusEq"; break;
        case sys::Token::MulEq:   ss << "MulEq";   break;
        case sys::Token::DivEq:   ss << "DivEq";   break;
        case sys::Token::ModEq:   ss << "ModEq";   break;
        case sys::Token::Le:      ss << "Le";      break;
        case sys::Token::Ge:      ss << "Ge";      break;
        case sys::Token::Gt:      ss << "Gt";      break;
        case sys::Token::Lt:      ss << "Lt";      break;
        case sys::Token::Eq:      ss << "Eq";      break;
        case sys::Token::Ne:      ss << "Ne";      break;
        case sys::Token::And:     ss << "And";     break;
        case sys::Token::Or:      ss << "Or";      break;
        case sys::Token::Semicolon: ss << "Semicolon"; break;
        case sys::Token::Assign:  ss << "Assign";  break;
        case sys::Token::Not:     ss << "Not";     break;
        case sys::Token::LPar:    ss << "LPar";    break;
        case sys::Token::RPar:    ss << "RPar";    break;
        case sys::Token::LBrak:   ss << "LBrak";   break;
        case sys::Token::RBrak:   ss << "RBrak";   break;
        case sys::Token::LBrace:  ss << "LBrace";  break;
        case sys::Token::RBrace:  ss << "RBrace";  break;
        case sys::Token::Comma:   ss << "Comma";   break;
        case sys::Token::End:     ss << "End";     break;
        default:                  ss << "UNKNOWN(" << tok.type << ")";
    }

    // 2. 如果 Token 有值，打印它的值
    switch (tok.type) {
        case sys::Token::LInt:
            ss << " [value: " << tok.vi << "]";
            break;
        case sys::Token::LFloat:
            ss << " [value: " << tok.vf << "]";
            break;
        case sys::Token::Ident:
            ss << " [value: " << tok.vs << "]";
            break;
        default:
            // 其他 Token 没有值
            break;
    }
    
    return ss.str();
}

// 你的测试主函数
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename.sy>" << std::endl;
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

    // 创建 Lexer 实例
    sys::Lexer lexer(source_code);

    std::cout << "--- Testing Lexer on file: " << filename << " ---" << std::endl;

    // 循环调用 nextToken()
    while (lexer.hasMore()) {
        sys::Token tok = lexer.nextToken();
        
        // 打印 Token
        std::cout << tokenToString(tok) << std::endl;
        
        if (tok.type == sys::Token::End) {
            break;
        }
    }

    std::cout << "--- Lexer Test Finished ---" << std::endl;

    // !!! 重要：测试你的内存释放 !!!
    // 你需要手动遍历 `tokens` 来释放 `vs`。
    // 但是在这个简单的测试驱动中，我们没有存储 Token。
    // 你可以在 Parser::parse() 中测试内存释放。

    return 0;
}