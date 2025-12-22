#include "Matcher.h"
#include "../codegen/Attrs.h"

using namespace sys;

// elements[0] is the operator name.
#define  MATCH_TERNARY(opcode, Ty) \
    if (opname == opcode && isa<Ty>(op)) { \
        return matchExpr(list->elements[1], op->getOperand(0).defining) && \
               matchExpr(list->elements[2], op->getOperand(1).defining) && \
               matchExpr(list->elements[3], op->getOperand(2).defining); \
    }

#define  MATCH_BINARY(opcode, Ty) \
    if (opname == opcode && isa<Ty>(op)) { \
        return matchExpr(list->elements[1], op->getOperand(0).defining) && \
               matchExpr(list->elements[2], op->getOperand(1).defining); \
    }

#define  MATCH_UNARY(opcode, Ty) \
    if (opname == opcode && isa<Ty>(op)) { \
        return matchExpr(list->elements[1], op->getOperand(0).defining); \
    }

bool Rule::matchExpr(Expr *expr, Op *op) {
    if (auto *atom = dyn_cast<Atom>(expr)) {
        std::string_view var = atom->value;
        if (var[0] == '*') {
            if (!isa<FloatOp>(op))
                return false;
        
            if (std::isdigit(var[1]) || var[1] == '-') {
                std::string str(var.substr(1));
                if (std::stof(str) != F(op))
                    return false;
            }

            if (binding.count(var))
                return F(binding[var]) == F(op);

            binding[var] = op;
            return true;
        }

        if (var[0] != '\'' && !(std::isdigit(var[0]) || var[0] == '-')) {
            if (binding.count(var))
                return binding[var] == op;
            binding[var] = op;
            return true;
        }

        if (!isa<IntOp>(op)) {
            return false;
        }

        if (std::isdigit(var[0]) || var[0] == '-') {
            std::string str(var);
            if (std::stoi(str) != V(op))
                return false;
        }

        if (binding.count(var))
            return V(binding[var]) == V(op);

        binding[var] = op;
        return true;
    }

    List *list = dyn_cast<List>(expr);
    if (!list)
        return false;

    assert(!list->elements.empty());

    Atom *head = dyn_cast<Atom>(list->elements[0]);
    if (!head)
        return false;

    std::string_view opname = head->value;

    MATCH_BINARY("lt", LtOp);
    MATCH_BINARY("le", LeOp);
    MATCH_UNARY("load", LoadOp);
    MATCH_BINARY("store", StoreOp);
    MATCH_BINARY("add", AddIOp);

    return false;
}

Rule::Rule(const char *text): text(text) {
    pattern = parse();
}

Rule::~Rule() {
    release(pattern);
}

void Rule::release(Expr *expr) {
    if (auto list = dyn_cast<List>(expr)) {
        for (auto elem : list->elements)
            release(elem);
    }
    delete expr;
}

void Rule::dump(std::ostream &os) {
    dump(pattern, os);
    os << "\n==== binding starts ====\n";
    for (auto &pair : binding) {
        os << pair.first << " = ";
        pair.second->dump(os);
        os << "\n";
    }
    os << "==== binding ends ====\n";
}

void Rule::dump(Expr *expr, std::ostream &os) {
    if (auto atom = dyn_cast<Atom>(expr)) {
        os << atom->value;
        return;
    }
    auto list = dyn_cast<List>(expr);
    os << "(";
    for (size_t i = 0; i < list->elements.size(); i++) {
        if (i > 0) os << " ";
        dump(list->elements[i], os);
    }
    os << ")";
}

std::string_view Rule::nextToken() {
    while (loc < text.size() && std::isspace(text[loc]))
        loc++;

    if (loc >= text.size())
        return "";

    if (text[loc] == '(' || text[loc] == ')')
        return text.substr(loc++, 1);

    int start = loc;
    while (loc < text.size() && !std::isspace(text[loc]) && text[loc] != '(' && text[loc] != ')')
        loc++;

    return text.substr(start, loc - start);
}

Expr *Rule::parse() {
    std::string_view tok = nextToken();
    // Each () matches a list.
    if (tok == "(") {
        auto list = new List();
        for(;;) {
            std::string_view peek = text.substr(loc, 1);
            if (peek == ")") {
                nextToken();
                break;
            }
            list->elements.push_back(parse());
        }
        return list;
    }
    return new Atom(tok);
}

bool Rule::match(Op *op, const std::map<std::string, Op*> &external) {
    loc = 0;
    failed = false;
    binding.clear();
    externalStrs.clear();
    for (auto &pair : external) {
        externalStrs.push_back(pair.first);
        binding[externalStrs.back()] = pair.second;
    }

    return matchExpr(pattern, op);
}

Op *Rule::extract(const std::string &name) {
    if (!binding.count(name)) {
        std::cerr << "querying unknown name: " << name << "\n";
        assert(false);
    }
    return binding[name];
}

