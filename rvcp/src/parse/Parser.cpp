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

ASTNode *Parser::primary() {
    if (peek(Token::LInt)) {
        int val = consume().vi;
        return new IntNode(val);
    }
    if (peek(Token::LFloat)) {
        float val = consume().vf;
        return new FloatNode(val);
    }
    if (test(Token::LPar)) {
        auto n = expr();
        expect(Token::RPar);
        return n;
    }

    if (peek(Token::Ident)) {
        auto vs = consume().vs;

        // Array access
        if (test(Token::LBrak)) {
            std::vector<ASTNode*> indices;
            do {
                indices.push_back(expr());
                expect(Token::RBrak);
            } while (test(Token::LBrak));
            
            if(test(Token::Assign)){
                // Array assign a[...] = x;
                auto value = expr();
                return new ArrayAssignNode(vs, indices, value);
            }
            // Array access x = a[...];
            return new ArrayAccessNode(vs, indices);
        }

        // Function call
        if (test(Token::LPar)) {
            std::vector<ASTNode*> args;
            while (!test(Token::RPar)) {
                args.push_back(expr());
                if (!test(Token::Comma) && !peek(Token::RPar))
                    expect(Token::RPar);
            }
            // Take special care for _sysy_{start,stop}time.
            // Their line numbers are encoded in their names.
            std::string name = vs;
            if (name.rfind("_sysy_starttime_", 0) != std::string::npos) {
                name = "_sysy_starttime";
                args.push_back(new IntNode(strtol(vs + 16, NULL, 10)));
            }
            if (name.rfind("_sysy_stoptime_", 0) != std::string::npos) {
                name = "_sysy_stoptime";
                args.push_back(new IntNode(strtol(vs + 15, NULL, 10)));
            }
            return new CallNode(name, args);
        }

        return new VarRefNode(vs);
    }
    std::cerr << "unexpected token " << peek().type << "\n";
    printSurrounding();
    assert(false);
}

ASTNode *Parser::unary() {
  if (test(Token::Minus))
    return new UnaryNode(UnaryNode::Minus, unary());

  if (test(Token::Plus))
    return unary();

  if (test(Token::Not))
    return new UnaryNode(UnaryNode::Not, unary());

  return primary();
}

ASTNode *Parser::mul() {
    auto n = unary();
    while (peek(Token::Mul, Token::Div, Token::Mod)) {
        switch (consume().type) {
        case Token::Mul:
            n = new BinaryNode(BinaryNode::Mul, n, unary());
            break;
        case Token::Div:
            n = new BinaryNode(BinaryNode::Div, n, unary());
            break;
        case Token::Mod:
            n = new BinaryNode(BinaryNode::Mod, n, unary());
            break;
        default:
            assert(false);
        }
    }
    return n;
}

ASTNode *Parser::add() {
    auto n = mul();
    while (peek(Token::Plus, Token::Minus)) {
        switch (consume().type) {
        case Token::Plus:
            n = new BinaryNode(BinaryNode::Add, n, mul());
            break;
        case Token::Minus:
            n = new BinaryNode(BinaryNode::Sub, n, mul());
            break;
        default:
            assert(false);
        }
    }
    return n;
}

ASTNode *Parser::rel() {
    auto n = add();
    while (peek(Token::Le, Token::Lt, Token::Ge, Token::Gt)) {
        switch (consume().type) {
        case Token::Le:
            n = new BinaryNode(BinaryNode::Le, n, add());
            break;
        case Token::Lt:
            n = new BinaryNode(BinaryNode::Lt, n, add());
            break;
        case Token::Ge:
            // a >= b  -> canonicalize to (b <= a)
            n = new BinaryNode(BinaryNode::Le, add(), n);
            break;
        case Token::Gt:
            // a > b  -> canonicalize to (b < a)
            n = new BinaryNode(BinaryNode::Lt, add(), n);
            break;
        default:
            assert(false);
        }
    }
    return n;
}

ASTNode *Parser::eq() {
    auto n = rel();
    while (peek(Token::Eq, Token::Ne)) {
        switch (consume().type) {
        case Token::Eq:
            n = new BinaryNode(BinaryNode::Eq, n, rel());
            break;
        case Token::Ne:
            n = new BinaryNode(BinaryNode::Ne, n, rel());
            break;
        default:
            assert(false);
        }
    }
    return n;
}

ASTNode *Parser::land() {
    auto n = eq();
    while (test(Token::And)) {
        n = new BinaryNode(BinaryNode::And, n, eq());
    }
    return n;
}

ASTNode *Parser::lor() {
    auto n = land();
    while (test(Token::Or)) {
        n = new BinaryNode(BinaryNode::Or, n, land());
    }
    return n;
}

ASTNode *Parser::expr() {
    return lor();
}

ASTNode *Parser::stmt() {
    if (test(Token::Semicolon)) {
        return new EmptyNode(); 
    }

    //debug: replace test with peek
    if (peek(Token::LBrace)) {
        return block();
    }

    if (test(Token::Return)) {
        if (test(Token::Semicolon))
            return new ReturnNode(currentFunc, nullptr);
        auto ret = new ReturnNode(currentFunc, expr());
        expect(Token::Semicolon);
        return ret;
    }

    if (test(Token::If)) {
        expect(Token::LPar);
        auto cond = expr();
        expect(Token::RPar);
        auto ifso = stmt();
        ASTNode *ifnot = nullptr;
        if (test(Token::Else)) {
            ifnot = stmt();
        }
        return new IfNode(cond, ifso, ifnot);
    }

    if (test(Token::While)) {
        expect(Token::LPar);
        auto cond = expr();
        expect(Token::RPar);
        auto body = stmt();
        return new WhileNode(cond, body);
    }

    if (test(Token::Break)) {
        expect(Token::Semicolon);
        return new BreakNode();
    }

    if (test(Token::Continue)) {
        expect(Token::Semicolon);
        return new ContinueNode();
    }

    if (test(Token::For)) {
        SemanticScope scope(*this);
        expect(Token::LPar);
        ASTNode *init = nullptr;
        if (peek(Token::Int) || peek(Token::Float) || peek(Token::Const)) {
            init = varDecl(false);
        } else if (!test(Token::Semicolon)) {
            auto n = expr();
            if (test(Token::Assign)) {
                if (!isa<VarRefNode>(n)) {
                    std::cerr << "expected lval in for-init\n";
                    assert(false);
                }
                auto value = expr();
                init = new AssignNode(n, value);
            } else {
                init = n;
            }
            expect(Token::Semicolon);
        }

        ASTNode *cond = nullptr;
        if (!test(Token::Semicolon)) {
            cond = expr();
            expect(Token::Semicolon);
        }

        ASTNode *incr = nullptr;
        if (!test(Token::RPar)) {
            auto n = expr();
            if (test(Token::Assign)) {
                if (!isa<VarRefNode>(n)) {
                    std::cerr << "expected lval in for-incr\n";
                    assert(false);
                }
                auto value = expr();
                incr = new AssignNode(n, value);
            } else {
                incr = n;
            }
            expect(Token::RPar);
        }

        auto body = stmt();

        return new ForNode(init, cond, incr, body);
    }

    if (peek(Token::Const, Token::Int, Token::Float))
    return varDecl(false);

    auto n = expr();
    if (test(Token::Assign)) {
        if (!isa<VarRefNode>(n)) {
        std::cerr << "expected lval\n";
        assert(false);
        }
        auto value = expr();
        expect(Token::Semicolon);
        return new AssignNode(n, value);
    }

    expect(Token::Semicolon);
    return n;
}

BlockNode *Parser::block() {
    SemanticScope scope(*this);
    expect(Token::LBrace);
    std::vector<ASTNode *> nodes;
    while (!test(Token::RBrace)) {
        nodes.push_back(stmt());
    }
    return new BlockNode(nodes);
}

TransparentBlockNode *Parser::varDecl(bool global) {
    bool mut = !test(Token::Const);
    auto base = parseSimpleType();
    std::vector<VarDeclNode*> decls;
    do {
        Type *ty= base;
        std::string name = expect(Token::Ident).vs;
        std::vector<int> dims;
        while (test(Token::LBrak)) {
            dims.push_back(earlyFold(expr()).getInt());
            expect(Token::RBrak);
        }
        if (dims.size() != 0)
            ty = new ArrayType(ty, dims);

        ASTNode *init = nullptr;
        // Array
        if (test(Token::Assign)) {
            bool isFloat = isa<FloatType>(base);
            if (isa<ArrayType>(ty)) {
                auto arrayInit = getArrayInit(dims, isFloat, global);
                init = !global
                    ? (ASTNode*) new LocalArrayNode((ASTNode **) arrayInit)
                    : isFloat
                        ? new ConstArrayNode((float*) arrayInit)
                        : new ConstArrayNode((int*) arrayInit);
                // We can never infer the real type of ConstArrayNode in Sema.
                // Must place it here.
                init->type = ty;
            }
            // simple
            else {
                init = expr();
                if (global)
                init = !isFloat
                    ? (ASTNode*) new IntNode(earlyFold(init).getInt())
                    : new FloatNode(earlyFold(init).getFloat());
            }
        }
        // No initialization; all must be zero.
        if (!init && global) {
            if (auto arrTy = dyn_cast<ArrayType>(ty)) {
                int size = arrTy->getSize();
                if (isa<FloatType>(base)) {
                    float *ptr = new float[size];
                    memset(ptr, 0, sizeof(float) * size);
                    init = new ConstArrayNode(ptr);
                } 
                else {
                    int *ptr = new int[size];
                    memset(ptr, 0, sizeof(int) * size);
                    init = new ConstArrayNode(ptr);
                }
                init->type = ty;
            } 
            else init = new IntNode(0);
        }

        auto decl = new VarDeclNode(name, init, mut, global);
        decl->type = ty;
        decls.push_back(decl);

        // Record in symbol table.
        if (!mut || global) {
            assert(init);
            symbols[name] = earlyFold(init);
        }

        if (!test(Token::Comma) && !peek(Token::Semicolon))
            expect(Token::Comma);
    } while (!test(Token::Semicolon));

    return new TransparentBlockNode(decls);
}

FnDeclNode *Parser::fnDecl() {
    Type *ret = parseSimpleType();
    auto name = expect(Token::Ident).vs;
    currentFunc = name;
    std::vector<Type *> params;
    std::vector<std::string> args;

    expect(Token::LPar);
    while(!test(Token::RPar)) {
        auto ty = parseSimpleType();
        args.push_back(expect(Token::Ident).vs);
        bool isPointer = false;
        if(test(Token::LBrak)) {
            isPointer = true;
            expect(Token::RBrak);
        }

        std::vector<int> dims;
        while(test(Token::LBrak)) {
            dims.push_back(earlyFold(expr()).getInt());
            expect(Token::RBrak);
        }
        
        if (dims.size()) {
            ty = ctx.create<ArrayType>(ty, dims);
        }

        if (isPointer) {
            ty = ctx.create<PointerType>(ty);
        }

        params.push_back(ty);
        if(!test(Token::Comma) && !peek(Token::RPar))
            expect(Token::Comma);
    }
    auto decl = new FnDeclNode(name, args, block());
    decl->type = ctx.create<FunctionType>(ret, params);
    return decl;
}

BlockNode *Parser::compUnit() {
    std::vector<ASTNode*> nodes;

    while (!test(Token::End)) {
        if (peek(Token::Const)) {
            //global
            nodes.push_back(varDecl(true));
            continue;
        }

        // For functions, it would be:
        //   Type ident `(`
        // while for variables it's `=`.
        // Moreover, the Type is only a single token,
        // so we lookahead for 2 tokens.
        if (tokens[loc + 2].type == Token::LPar) {
            nodes.push_back(fnDecl());
            continue;
        }

        nodes.push_back(varDecl(true));
    }

    return new BlockNode(nodes);
}

// Yes, heavy memory leak... But who cares?
// We can't just call release(), otherwise we'd release everything in the symbol table.
ConstValue Parser::earlyFold(ASTNode *node) {
  if (auto ref = dyn_cast<VarRefNode>(node)) {
    if (!symbols.count(ref->name)) {
      std::cerr << "cannot find const: " << ref->name << "\n";
      assert(false);
    }
    return symbols[ref->name];
  }

  if (auto binary = dyn_cast<BinaryNode>(node)) {
    auto lv = earlyFold(binary->l);
    auto rv = earlyFold(binary->r);
    if (!lv.isFloat && !rv.isFloat) {
      int l = lv.getInt(), r = rv.getInt();
      switch (binary->kind) {
      case BinaryNode::Add:
        return ConstValue(new int(l + r), {});
      case BinaryNode::Sub:
        return ConstValue(new int(l - r), {});
      case BinaryNode::Mul:
        return ConstValue(new int(l * r), {});
      case BinaryNode::Div:
        // Divide by zero might get generated by fuzzer
        return ConstValue(new int(r ? l / r : 0), {});
      case BinaryNode::Mod:
        return ConstValue(new int(r ? l % r : 0), {});
      case BinaryNode::And:
        return ConstValue(new int(l && r), {});
      case BinaryNode::Or:
        return ConstValue(new int(l || r), {});
      case BinaryNode::Eq:
        return ConstValue(new int(l == r), {});
      case BinaryNode::Ne:
        return ConstValue(new int(l != r), {});
      case BinaryNode::Lt:
        return ConstValue(new int(l < r), {});
      case BinaryNode::Le:
        return ConstValue(new int(l > r), {});
      }
    }
    
    // Implicitly raise to float.
    float l = lv.isFloat ? lv.getFloat() : lv.getInt();
    float r = rv.isFloat ? rv.getFloat() : rv.getInt();
    switch (binary->kind) {
    case BinaryNode::Add:
      return ConstValue(new float(l + r), {});
    case BinaryNode::Sub:
      return ConstValue(new float(l - r), {});
    case BinaryNode::Mul:
      return ConstValue(new float(l * r), {});
    case BinaryNode::Div:
      return ConstValue(new float(l / r), {});
    case BinaryNode::Mod:
      assert(false);
    // Note these relation operations should return int.
    case BinaryNode::And:
      return ConstValue(new int(l && r), {});
    case BinaryNode::Or:
      return ConstValue(new int(l || r), {});
    case BinaryNode::Eq:
      return ConstValue(new int(l == r), {});
    case BinaryNode::Ne:
      return ConstValue(new int(l != r), {});
    case BinaryNode::Lt:
      return ConstValue(new int(l < r), {});
    case BinaryNode::Le:
      return ConstValue(new int(l > r), {});
    }
  }

  if (auto unary = dyn_cast<UnaryNode>(node)) {
    auto v = earlyFold(unary->node);
    if (v.isFloat) {
      switch (unary->kind) {
      case UnaryNode::Minus:
        return ConstValue(new float(-v.getFloat()), {});
      // Note this should return int.
      case UnaryNode::Not:
        return ConstValue(new int(!v.getFloat()), {});
      default:
        assert(false);
      }
    }

    switch (unary->kind) {
    case UnaryNode::Minus:
      return ConstValue(new int(-v.getInt()), {});
    case UnaryNode::Not:
      return ConstValue(new int(!v.getInt()), {});
    default:
      assert(false);
    }
  }

  if (auto lint = dyn_cast<IntNode>(node))
    return ConstValue(new int(lint->value), {});

  if (auto lfloat = dyn_cast<FloatNode>(node))
    return ConstValue(new float(lfloat->value), {});

  if (auto access = dyn_cast<ArrayAccessNode>(node)) {
    if (!symbols.count(access->array)) {
      std::cerr << "cannot find constant: " << access->array << "\n";
      assert(false);
    }
    auto array = symbols[access->array];
    ConstValue v = array;
    for (auto index : access->indices)
      v = v[earlyFold(index).getInt()];
    return v;
  }

  if (auto arr = dyn_cast<ConstArrayNode>(node))
    return ConstValue(arr->vi, cast<ArrayType>(arr->type)->dims);
  
  if (auto arr = dyn_cast<LocalArrayNode>(node)) {
    // This implies that the whole LocalArray is constant. Try to fold it.
    auto arrTy = cast<ArrayType>(arr->type);
    bool isFloat = isa<FloatType>(arrTy->base);
    if (isFloat) {
      assert(false);
    } else {
      int size = arrTy->getSize();
      int *result = new int[size];
      for (int i = 0; i < size; i++) {
        auto node = arr->elements[i];
        if (!node) {
          result[i] = 0;
          continue;
        }

        result[i] = earlyFold(node).getInt();
      }
      return ConstValue(result, arrTy->dims);
    }
  }

  std::cerr << "not constexpr: " << node->getID() << "\n";
  assert(false);
}

Parser::Parser(const std::string &input, TypeContext &ctx): loc(0), ctx(ctx) {
  Lexer lex(input);

  while (lex.hasMore())
    tokens.push_back(lex.nextToken());
}

ASTNode *Parser::parse() {
  auto unit = compUnit();

  // Release memory.
  for (auto tok : tokens) {
    if (tok.type == Token::Ident)
      delete[] tok.vs;
  }

  return unit;
}
