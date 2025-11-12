#include "Parser.h"
#include "ASTNode.h"
#include "Lexer.h"
#include "Type.h"
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
    assert(i < dims[0]);
    std::vector<int> newDims.reserve(dims.size() - 1);
    for (int j = 1; j < dims.size(); j++) {
        newDims.push_back(dims[j]);
    }
    return ConstValue(vi + i * stride(), newDims);
}

