#include "Type.h"
#include <sstream>

using namespace sys;

std::string interleave(const std::vector<Type*> &types) {
    std::ostringstream oss;
    for (size_t i = 0; i < types.size(); ++i) {
        oss << types[i]->toString();
        if (i != types.size() - 1) {
            oss << ",";
        }
    }
    return oss.str();
}

std::string FunctionType::toString() const {
  return "(" + interleave(params) + ") -> " + ret->toString();
}

std::string ArrayType::toString() const {
  std::stringstream oss;
  oss << base->toString();
  for (auto x : dims)
    oss << "[" << x << "]";
  return oss.str();
}

// not byte size, but number of elements
int ArrayType::getSize() const {
  int size = 1;
  for (auto x : dims)
    size *= x;
  return size;

};