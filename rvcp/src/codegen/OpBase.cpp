#include "OpBase.h"
#include "Attrs.h"
#include "Ops.h"


using namespace sys;

std::map<BasicBlock*, int> sys::bbmap;
int sys::bbid = 0;

static int getBlockID(BasicBlock *bb) {
  if (!bbmap.count(bb))
    bbmap[bb] = bbid++;
  return bbmap[bb];
}

void BasicBlock::insert(iterator at, Op *op) {
  op->parent = this;
  op->place = ops.insert(at, op);
}

void BasicBlock::insertAfter(iterator at, Op *op) {
  op->parent = this;
  // insert before std::list::end() iterator
  if (at == ops.end()) {
    ops.push_back(op);
    op->place = --end();
    return;
  }
  op->place = ops.insert(++at, op);
}

void BasicBlock::remove(iterator at) {
  ops.erase(at);
}

BasicBlock* BasicBlock::nextBlock() const {
    auto it = place;
    return *++it;
}

Op* Op::prevOp() {
    auto it = place;
    if (it == parent->begin())
        return nullptr;
    return *--it;
}

Op* Op::nextOp() {
    auto it = place;
    if (++it == parent->end())
        return nullptr;
    return *it;
}

Value::Value(Op *from): defining(from) {}

Op::Op(int id, Value::Type resulty, const std::vector<Value> &values):
    resultTy(resultTy), opid(id) {
    for (auto x : values) {
        operands.push_back(x);
        x.defining->uses.insert(this);
    }
}

Op::Op(int id, Value::Type resultTy, const std::vector<Value> &values, const std::vector<Attr*> &attrs):
  resultTy(resultTy), opid(id) {
  for (auto x : values) {
    operands.push_back(x);
    x.defining->uses.insert(this);
  }
  for (auto attr : attrs) {
    auto cloned = attr->clone();
    this->attrs.push_back(cloned);
    cloned->refcnt++;
    if (!attr->refcnt)
      delete attr;
  }
}

void Op::setName(std::string name) {
  // Remove "Op" suffix.
  name.pop_back();
  name.pop_back();
  for (auto &c : name)
    c = tolower(c);
  opname = name;
}

// space indentation
void indent(std::ostream &os, int n) {
  for (int i = 0; i < n; i++)
    os << ' ';
}

Region* Op::appendRegion() {
  auto region = new Region(this);
  regions.push_back(region);
  return region;
}

void Op::pushOperand(Value v) {
    v.defining->uses.insert(this);
    operands.push_back(v);
}

Op* Op::getParentOp() {
    auto bb = parent;
    auto region = bb->parent;
    return region->getParent();
}

void Op::removeAllOperands() {
    for (auto x : operands) {
        auto op = x.defining;
        op->uses.erase(this);
    }
    operands.clear();
}

void Op::removeOperand(int i) {
    auto def = operands[i].defining;
    operands.erase(operands.begin() + i);
    removeOperandUse(def);
}

void Op::removeOperandUse(Op *def) {
    bool hasDef = false;
    for (auto x : operands) {
        if (x.defining == def) {
            hasDef = true;
            break;
        }
    }
    // check if we still refer to def. If not, remove it from def's uses.
    if (!hasDef)
        def->uses.erase(this);
}

void Op::removeOperand(Op *v) {
    for (int i = 0; i < operands.size(); i++) {
        auto def = operands[i].defining;
        if (def == v) {
            removeOperand(i);
            return;
        }
    }
    assert(false);
}

void Op::setOperand(int i, Value v) {
    auto def = operands[i].defining;
    operands[i] = v;
    removeOperandUse(def);
    v.defining->uses.insert(this);
}

// before is defining op used as operand
int Op::replaceOperand(Op *before, Value v) {
    for (int i = 0; i < operands.size(); i++) {
        auto def = operands[i].defining;
        if (def == before) {
            setOperand(i, v);
            return i;
        }
    }
    assert(false);
}

void Op::removeAllAttributes() {
    for (auto attr : attrs) {
        if (!--attr->refcnt)
            delete attr;
    }
    attrs.clear();
}

void Op::removeAttribute(int i){
    auto attr = attrs[i];
    if (!--attr->refcnt)
        delete attr;
    attrs.erase(attrs.begin() + i);
}

void Op::setAttribute(int i, Attr *attr) {
    attr->refcnt++;
    if (!--attrs[i]->refcnt)
        delete attrs[i];
    attrs[i] = attr;
}

void Op::removeRegion(Region *region){
    for (auto it = regions.begin(); it != regions.end(); it++) {
        if (*it == region) {
            regions.erase(it);
            break;
        }
    }
}

// only remove if we don't refer to it anymore.
void Op::erase() {
    parent->remove(place);
    removeAllOperands();

    for (auto region : regions)
        region->erase();
    // if used elsewhere, error out.
    if (uses.size()) {
        std::cerr << "removing op in use:\n  ";
        dump(std::cerr);
        std::cerr << "uses:\n";
        for (auto use : uses) {
            std::cerr << "  ";
            use->dump(std::cerr);
        }
        assert(false);
    }
    toDelete.push_back(this);
}

std::vector<Op*> Op::toDelete;

// release all the ops that are marked to be deleted
void Op::release() {
    for (auto op : toDelete) {
        for (auto attr : op->attrs) {
            if (!--attr->refcnt)
                delete attr;
        }
        delete op;
    }
    toDelete.clear();
}

BasicBlock* Op::createFirstBlock() {
    appendRegion();
    return regions[0]->appendBlock();
}

void Op::replaceAllUsesWith(Op *other){
    for (auto use : uses) {
        for (auto &operand : use->operands) {
            if (operand.defining != this)
                continue;

            operand.defining = other;
            other->uses.insert(use);
        }
    }
    uses.clear();
}

static std::map<Op*, int> valueName = {};
static int id = 0;

std::string getValueNumber(Value value) {
    if (!valueName.count(value.defining))
        valueName[value.defining] = id++;
    return "%" + std::to_string(valueName[value.defining]);
}

// Op dump
void Op::dump(std::ostream &os, int depth) {
    indent(os, depth * 2);
    os << getValueNumber(getResult()) << " = " << opname;
    if (resultTy == Value::f32)
        os << ".f";
    for (auto &operand : operands)
        os << " " << getValueNumber(operand);
    for (auto attr : attrs)
        os << " " << attr->toString();
    if (regions.size() > 0) {
        os << " ";
        for (auto &region : regions)
        region->dump(os, depth + 1);
    }
    os << "\n";
}

void Op::moveBefore(Op *op) {
    if (op == this)
        return;
    parent->remove(place);
    parent = op->parent;
    parent->insert(op->place, this);
}

void Op::moveAfter(Op *op) {
    if (op == this)
        return;
    parent->remove(place);
    parent = op->parent;
    parent->insertAfter(op->place, this);
}

// move to the end of the given basicblock
void Op::moveToEnd(BasicBlock *bb) {
    parent->remove(place);
    parent = bb;
    parent->insert(parent->end(), this);
}

void Op::moveToStart(BasicBlock *bb) {
    parent->remove(place);
    parent = bb;
    parent->insert(parent->begin(), this);
}

Op *Op::getPhiFrom(Op * phi, BasicBlock *bb) {
    const auto &ops = phi->operands;
    const auto &attrs = phi->attrs;
    for (int i = 0; i < ops.size(); i++) {
        // if (cast<FromAttr>(attrs[i])->bb == bb)
        if (FROM(attrs[i]) == bb)
            return phi->DEF(i);
    }

    std::cerr << "no operand from bb." << bbmap[bb] << ": " << phi;
    assert(false);
}

BasicBlock *Op::getPhiFrom(Op* phi, Op* op) {
    const auto &ops = phi->operands;
    const auto &attrs = phi->attrs;
    for (int i = 0; i < ops.size(); i++) {
        if (ops[i].defining == op)
            return FROM(attrs[i]);
    }
    assert(false);
}

bool Op::inside(Op *op) {
    for (Op *runner = this; !isa<ModuleOp>(runner); runner = runner->getParentOp()) {
        if (op == runner)
            return true;
    }
    return false;
}

std::vector<Op*> BasicBlock::getPhis() const {
    std::vector<Op*> phis;
    for (auto op : ops) {
        if (!isa<PhiOp>(op))
            break;

        phis.push_back(op);
    }
    return phis;
}

bool BasicBlock::dominatedBy(const BasicBlock *bb) const {
    for (auto p = this; p; p = p->idom) {
        if (p == bb)
            return true;
    }
    return false;
}

// it++ trick to avoid iterator invalidation.
void BasicBlock::inlineBefore(Op *op) {
    for (auto it = begin(); it != end(); ) {
        auto next = it; ++next;
        (*it)->moveBefore(op);
        it = next;
    }
}

void BasicBlock::inlineToEnd(BasicBlock *bb) {
    for (auto it = begin(); it != end(); ) {
        auto next = it; ++next;
        (*it)->moveToEnd(bb);
        it = next;
    }
}

void BasicBlock::splitOpsAfter(BasicBlock *dest, Op *op) {
    for (auto it = op->place; it != end(); ) {
        auto next = it; ++next;
        (*it)->moveToEnd(dest);
        it = next;
    }
}

void BasicBlock::splitOpsBefore(BasicBlock *dest, Op *op) {
  for (auto it = begin(); it != op->place; ) {
    auto next = it; ++next;
    (*it)->moveToEnd(dest);
    it = next;
  }
}

void BasicBlock::moveBefore(BasicBlock *bb) {
    parent->remove(place);
    parent = bb->parent;
    parent->insert(bb->place, this);
}

void BasicBlock::moveAfter(BasicBlock *bb) {
    parent->remove(place);
    parent = bb->parent;
    parent->insertAfter(bb->place, this);
}

void BasicBlock::moveToEnd(Region *region) {
    parent->remove(place);
    parent = region;
    parent->insert(parent->end(), this);
}

void BasicBlock::erase() {
    if (preds.size() != 0) {
        std::cerr << "Erasing block with preds!\nself = bb" << bbmap[this] << "; all preds: ";
        for (auto pred : preds) {
            std::cerr << "bb" << bbmap[pred] << " ";
        }
        std::cerr << "\n";
        assert(false);
    }
    forceErase();
}

void BasicBlock::forceErase() {
    auto copy = ops;
    for (auto op : copy) 
        op->removeAllOperands();
    for (auto op : copy) 
        op->erase();

    parent->remove(place);
    delete this;
}

BasicBlock* Region::insert(BasicBlock* at) {
    assert(at->parent == this);

    auto it = bbs.insert(at->place, nullptr);
    *it = new BasicBlock(this, it);
    return *it;
}

BasicBlock* Region::insertAfter(BasicBlock* at) {
    assert(at->parent == this);

    if (at->place == end())
        return appendBlock();

    auto place = at->place;
    ++place;
    auto it = bbs.insert(place, nullptr);
    *it = new BasicBlock(this, it);
    return *it;
}

void Region::remove(BasicBlock* at) {
    bbs.erase(at->place);
}

void Region::remove(iterator at) {
    bbs.erase(at);
}

void Region::insert(iterator at, BasicBlock *bb) {
    bb->parent = this;
    bb->place = bbs.insert(at, bb);
}

void Region::insertAfter(iterator at, BasicBlock *bb) {
    bb->parent = this;
    // insert before std::list::end() iterator
    if (at == bbs.end()) {
        bbs.push_back(bb);
        bb->place = --end();
        return;
    }
    auto place = at;
    ++place;
    bb->place = bbs.insert(place, bb);
}

BasicBlock* Region::appendBlock() {
    bbs.push_back(nullptr);
    auto place = --bbs.end();
    *place = new BasicBlock(this, place);
    return *place;
}

std::pair<BasicBlock*, BasicBlock*> Region::moveTo(BasicBlock *bb) {
    BasicBlock *prev = bb;
    // Preserve it beforehand; the region will become empty afterwards.
    auto result = std::make_pair(getFirstBlock(), getLastBlock());
    for (auto it = begin(); it != end();) {
        auto next = it; ++next;
        auto current = *it;
        current->moveAfter(prev);
        prev = current;
        it = next;
    }
    return result;
}

void Region::erase() {
    auto copy = bbs;
    for (auto bb : copy)
        bb->forceErase();
    parent->removeRegion(this);
    delete this;
}

// step1: clear all preds and succs
// step2: for each block, look at its terminator to find its targets, and
//         add itself to the target's preds
// step3: for each block, for each of its preds, add the pred to its succs
void Region::updatePreds() {
    for (auto bb : bbs) {
        bb->preds.clear();
        bb->succs.clear();
    }
    for (auto bb : bbs) {
        assert(bb->getOpCount() > 0);
        auto last = bb->getLastOp();

        if (last->has<TargetAttr>()) {
            auto target = last->get<TargetAttr>();
            target->bb->preds.insert(bb);
        }

        if (last->has<ElseAttr>()) {
            auto ifnot = last->get<ElseAttr>();
            ifnot->bb->preds.insert(bb);
        }
    }

    for (auto bb : bbs) {
        for (auto pred : bb->preds) {
            bb->succs.insert(pred);
        }
    }
}


