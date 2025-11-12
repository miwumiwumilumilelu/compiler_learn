#ifndef LEXER_H
#define LEXER_H

#include <string>
#include <vector>
#include <cstring>

namespace sys {

struct Token{
    enum Type {
        // Literals
        LInt, LFloat, Ident,

        // Keywords
        If, Else, While, Return, For, Int, Float, Void,
        Const, Break, Continue,
            
        // Operators
        Minus, Plus, Mul, Div, Mod,
        PlusEq, MinusEq, MulEq, DivEq, ModEq,
        Le, Ge, Gt, Lt, Eq, Ne,
        And, Or, Semicolon, Assign, Not,
        LPar, RPar, LBrak, RBrak, LBrace, RBrace,
        Comma,

        //End
        End,
    }type;

    // have to manually free space anyway (Destructors won't be called in a union)
    union {
        int vi;
        float vf;
        char *vs;
    };
        
    // Constructors
    /* implicit */ Token(Type t): type(t) {}
    /* implicit */ Token(int vi): type(LInt), vi(vi) {}
    /* implicit */ Token(float vf): type(LFloat), vf(vf) {}
    /* implicit */ Token(const std::string &str): type(Ident), vs(new char[str.size() + 1]) {
    strcpy(vs, str.c_str());
    }
};

class Lexer {
  std::string input;

  // Index of `input`
  size_t loc = 0;
  size_t lineno = 1;
public:
  Lexer(const std::string &input): input(input) {}

  Token nextToken();
  bool hasMore() const;
};

}

#endif // LEXER_H