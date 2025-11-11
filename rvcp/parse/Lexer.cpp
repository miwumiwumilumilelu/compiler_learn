#include "Lexer.h"
#include <cassert>
#include <cctype>
#include <cmath>
#include <map>

using namespace sys;

std::map<std::string, Token::Type> keywords = {
    {"if", Token::If},
    {"else", Token::Else},
    {"while", Token::While},
    {"return", Token::Return},
    {"for", Token::For},
    {"int", Token::Int},
    {"float", Token::Float},
    {"void", Token::Void},
    {"const", Token::Const},
    {"break", Token::Break},
    {"continue", Token::Continue},
};

bool Lexer::hasMore() const {
    return loc < input.size();
}

Token Lexer::nextToken() {
    while (hasMore() && std::isspace(input[loc])) {
        if(input[loc] == '\n'){
            lineno++;
        }
        loc++;
    }

    // Hit end of input because of skipping whitespace
    if(loc >= input.size()){
        return Token::End;
    }

    char c = input[loc];

    // Identifiers and keywords
    if (std::isalpha(c) || c == '_') {
        std::string name;

        while(loc < input.size() && (std::isalnum(input[loc]) || input[loc] == '_')){
            name += input[loc++];
        }

        if(keywords.count(name)){
            return Token(keywords[name]);
        }

        if (name == "stoptime")
            return Token("_sysy_stoptime_" + std::to_string(lineno));
        if (name == "starttime")
            return Token("_sysy_starttime_" + std::to_string(lineno));

        return Token(name);
    }

    // Integer literals
    if (std::isdigit(c) || c == '.') {
        int start = loc;
        bool isFloat = false;

        // Skip '0x' or '0X'
        if (c == '0' && (input[loc + 1] == 'x' || input[loc + 1] == 'X')) {
            loc += 2; 
            while(input[loc] == '.' || std::isxdigit(input[loc])){
                if (input[loc] == '.') {
                    if (isFloat) break; // Second decimal point
                    isFloat = true;
                }
                loc++;
            }

            // p or P for hexadecimal floating-point exponent
            if (input[loc] == 'p' || input[loc] == 'P') {
                isFloat = true;
                loc++;
                // 0x1.Ap-2
                if (input[loc] == '+' || input[loc] == '-') {
                    loc++;
                }
                while(std::isdigit(input[loc])){
                    loc++;
                }
            }
            
            // Convert the raw string to int or float
            std::string raw = input.substr(start, loc - start);
            return isFloat ? Token(strtof(raw.c_str(), nullptr)) : std::stoi(raw, nullptr, /*base = autodetect*/0);
        }

        // normal decimal number
        while(loc < input.size() && (std::isdigit(input[loc]) || input[loc] == '.')){
            if (input[loc] == '.') {
                if (isFloat) break; // Second decimal point
                isFloat = true;
            }
            loc++;
        }

        if (input[loc] == 'e' || input[loc] =='E'){
            isFloat = true;
            loc++;
            if(input[loc] == '+' || input[loc] == '-') {
                loc++;
            }
            while(std::isdigit(input[loc])){
                loc++;
            }
        }

        std::string raw = input.substr(start, loc - start);
        return isFloat ? Token(strtof(raw.c_str(), nullptr)) : std::stoi(raw, nullptr, /*base = autodetect*/0);
    }

    // Operators and punctuation
    switch (c) {
        case '=': 
            if (input[loc + 1] == '=') { loc += 2; return Token::Eq; }
            break;
        case '>':
            if (input[loc + 1] == '=') { loc += 2; return Token::Ge; }
            break;
        case '<': 
            if (input[loc + 1] == '=') { loc += 2; return Token::Le; }
            break;
        case '!': 
            if (input[loc + 1] == '=') { loc += 2; return Token::Ne; }
            break;
        case '+': 
            if (input[loc + 1] == '=') { loc += 2; return Token::PlusEq; }
            break;
        case '-': 
            if (input[loc + 1] == '=') { loc += 2; return Token::MinusEq; }
            break;
        case '*': 
            if (input[loc + 1] == '=') { loc += 2; return Token::MulEq; }
            break;
        case '/':
            if (input[loc + 1] == '=') {
                loc += 2;
                return Token::DivEq;
            } else if (input[loc + 1] == '/') {
                // Single-line comment
                for(; loc < input.size(); loc++){
                    if(input[loc] == '\n'){
                        return nextToken();
                    }
                }
                return Token::End;
            } else if (input[loc + 1] == '*') {
                // Multi-line comment
                loc += 2;
                for(; loc + 1 < input.size(); loc++){
                    if(input[loc] == '*' && input[loc + 1] == '/'){
                        loc += 2; // Skip '*/'
                        return nextToken();
                    }
                }
                return Token::End;
            }
            break;
        case '%': 
            if (input[loc + 1] == '=') { loc += 2; return Token::ModEq; }
            break;
        case '&': 
            if (input[loc + 1] == '&') { loc += 2; return Token::And; }
            break;
        case '|': 
            if (input[loc + 1] == '|') { loc += 2; return Token::Or; }
            break;
        default:
            break;
        }
    
    switch (c) {
        case '-': loc++; return Token::Minus;
        case '+': loc++; return Token::Plus;
        case '*': loc++; return Token::Mul;
        case '/': loc++; return Token::Div;
        case '%': loc++; return Token::Mod;
        case ';': loc++; return Token::Semicolon;
        case '=': loc++; return Token::Assign;
        case '!': loc++; return Token::Not;
        case '(': loc++; return Token::LPar;
        case ')': loc++; return Token::RPar;
        case '[': loc++; return Token::LBrak;
        case ']': loc++; return Token::RBrak;
        case '{': loc++; return Token::LBrace;
        case '}': loc++; return Token::RBrace;
        case ',': loc++; return Token::Comma;
        case '>': loc++; return Token::Gt;
        case '<': loc++; return Token::Lt;
        default:
            // Unknown character
            assert(false && "Unknown character in input");
    }

}
