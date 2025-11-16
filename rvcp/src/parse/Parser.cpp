#include "Parser.h"
#include "ASTNode.h"
#include "Lexer.h"

#include "TypeContext.h"
#include <vector>
#include <cassert>
#include <ostream>

using namespace sys;

int ConstValue::size() {
    int s = 1;
    for (auto d : dims) {
        s *= d;
    }
    return s;
}

int ConstValue::stride(){
    if (dims.size() <= 1) return 1;
    int s = 1;
    for (int i = 1; i < dims.size(); i++) {
        s *= dims[i];
    }
    return s;
}

std::ostream& operator<<(std::ostream &os, ConstValue cv){
    auto sz = cv.size();
    auto vi = (int *)cv.getRawRef();
    os << vi[0];
    for(int i=1; i<sz; i++){
        os << "," << vi[i];
    }
    return os;
}

std::ostream& operator<<(std::ostream &os, const std::vector<int> vec){
    if (vec.size() > 0)
        os << vec[0];
    for (int i = 1; i < vec.size(); i++)
        os << ", " << vec[i];
    return os;
}

ConstValue ConstValue::operator[](int i) {
    assert(dims.size() >= 1);
    std::vector<int> newDims;
    newDims.reserve(dims.size() - 1);

    for (int j = 1; j < dims.size(); j++) {
        newDims.push_back(dims[j]);
    }

    return ConstValue(vi + i * stride(), newDims);
}

int *ConstValue::getRaw() {
    int sz = size();
    int *newMem = new int[sz];
    memcpy(newMem, vi, sz * sizeof(int));
    return newMem;
}

float *ConstValue::getRawFloat() {
    int sz =size();
    float *newMem = new float[sz];
    memcpy(newMem, vf ,sz * sizeof(float));
    return newMem;
}

void ConstValue::release() {
    if (isFloat) {
        delete[] vf;
        vf = nullptr;
    } else {
        delete[] vi;
        vi = nullptr;
    }
}

int ConstValue::getInt() {
    assert((!dims.size() || (dims[0] == 1 && dims.size() == 1)));
    if(isFloat){
        return *vf;
    }
    return *vi;
}

float ConstValue::getFloat() {
    assert((!dims.size() || (dims[0] == 1 && dims.size() == 1)));
    if(!isFloat){
        return *vi;
    }
    return *vf;
}

Token Parser::last() {
    if(loc-1 >= tokens.size()){
        return Token::End;
    }
    return tokens[loc-1];
}

Token Parser::peek() {
    if(loc >= tokens.size()){
        return Token::End;
    }
    return tokens[loc];
}   

Token Parser::consume() {
  if (loc >= tokens.size()) {
      return Token::End;
  }
  return tokens[loc++];
}

bool Parser::peek(Token::Type t) {
    return peek().type == t;
}

Token Parser::expect(Token::Type t) {
  if (!test(t)) {
    std::cerr << "expected " << t << ", but got " << peek().type << "\n";
    printSurrounding();
    assert(false);
  }
  return last();
}

// Prints tokens in range [loc-5, loc+5]. For debugging purposes.
void Parser::printSurrounding() {
  std::cerr << "surrounding:\n";
  for (size_t i = std::max(0ul, loc - 5); i < std::min(tokens.size(), loc + 6); i++) {
    std::cerr << tokens[i].type;
    if (tokens[i].type == Token::LInt) {
      std::cerr << " <int = " << tokens[i].vi << ">";
    }
    if (tokens[i].type == Token::LFloat) {
      std::cerr << " <float = " << tokens[i].vf << "f>";
    }
    if (tokens[i].type == Token::Ident) {
      std::cerr << " <name = " << tokens[i].vs << ">";
    }
    std::cerr << (i == loc ? " (here)" : "") << "\n";
  }
}

Type *Parser::parseSimpleType() {
    if (test(Token::Int)) {
        return ctx.create<IntType>();
    } else if (test(Token::Float)) {
        return ctx.create<FloatType>();
    } else if (test(Token::Void)) {
        return ctx.create<VoidType>();
    } else {
        std::cerr << "expected type, but got " << peek().type << "\n";
        printSurrounding();
        assert(false);
    }
}

void *Parser::getArrayInit(const std::vector<int> &dims, bool expectFloat, bool doFold) {
    auto carry = [&](std::vector<int> &x) {
        for(int i = (int) x.size() - 1 ; i>=1 ; i--){
            if (x[i] >= dims[i]){
                auto quto = x[i] / dims[i];
                x[i] = x[i] % dims[i];
                x[i-1] += quto;
            }
        }
    };

    auto offset = [&](std::vector<int> &x) {
        int total = 0, stride = 1;
        for (int i = (int) x.size() - 1; i >= 0; i--) {
        total += x[i] * stride;
        stride *= dims[i];
        }
        return total;
    }; 

    // initialize with 'dims.size()' zeroes.
    std::vector<int> place(dims.size(), 0);
    int size = 1;
    for (auto x : dims)
        size *= x;
    
    void *vi = !doFold
        ? (void*) new ASTNode*[size]
        : expectFloat ? (void*) new float[size] : new int[size];
    
    memset(vi, 0, size * (doFold ? expectFloat ? sizeof(float) : sizeof(int) : sizeof(ASTNode*)));

      // add 1 to `place[addAt]` when we meet the next `}`.
    int addAt = -1;
    do {
        if (test(Token::LBrace)) {
        addAt++;
        continue;
        }

        if (test(Token::RBrace)) {
            if (--addAt == -1)
                break;

            // Bump `place[addAt]`, and set everything after it to 0.
            place[addAt]++;
            for (int i = addAt + 1; i < dims.size(); i++)
                place[i] = 0;
            if (!peek(Token::RBrace))
                carry(place);
            
            // If this `}` isn't at the end, then a `,` or `}` must follow.
            if (addAt != -1 && !peek(Token::RBrace))
                expect(Token::Comma);
            continue;
        }

        if (!doFold)
            ((ASTNode**) vi)[offset(place)] = expr();
        else if (expectFloat)
            ((float*) vi)[offset(place)] = earlyFold(expr()).getFloat();
        else
            ((int*) vi)[offset(place)] = earlyFold(expr()).getInt();

        place[place.size() - 1]++;

        // Automatically carry.
        // But don't carry if the next token is `}`.
        if (!peek(Token::RBrace))
            carry(place);
        if (!test(Token::Comma) && !peek(Token::RBrace))
            expect(Token::RBrace);
    } while (addAt != -1);

return vi;  
}

