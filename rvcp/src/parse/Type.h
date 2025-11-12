#ifndef TYPE_H
#define TYPE_H

#include <string>
#include <vector>

namespace sys {

class Type {
    const int id;
public:
    int getID() const { return id; }
    virtual std::string toString() const = 0;
    virtual ~Type() {}
    Type(int id): id(id) {}
};

template<class T, int TypeID>
class TypeImpl : public Type {
public:
  static bool classof(Type *ty) {
    return ty->getID() == TypeID;
  }

  TypeImpl(): Type(TypeID) {}
};


}

#endif // TYPE_H