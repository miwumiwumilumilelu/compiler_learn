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
    resultTy(resulty), opid(id) {
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

    auto regionsCopy = regions;
    for (auto region : regionsCopy)
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
            pred->succs.insert(bb);
        }
    }
}

namespace {
// DFN is the number of each node in DFS order.
using DFN = std::unordered_map<BasicBlock*, int>;
using BBMap = std::unordered_map<BasicBlock*, BasicBlock*>;

using Vertex = std::vector<BasicBlock*>;

using SDom = BBMap;
using Parent = BBMap;
using UnionFind = BBMap;
using Best = BBMap;

int num = 0;
int pnum = 0;

// Dominators.
DFN dfn;
SDom sdom;
Vertex vertex;
Parent parents;
UnionFind uf;
Best best;
// Post-dominators. Just a copy-paste.
DFN pdfn;
SDom psdom;
Vertex pvertex;
Parent pparents;
UnionFind puf;
Best pbest;

void updateDFN(BasicBlock *current) {
    dfn[current] = num++;
    vertex.push_back(current);
    for (auto v : current->succs) {
        if (!dfn.count(v)) {
            parents[v] = current;
            updateDFN(v);
        }
    }
}

void updatePDFN(BasicBlock *current) {
    pdfn[current] = pnum++;
    pvertex.push_back(current);
    for (auto v : current->preds) {
        if (!pdfn.count(v)) {
        pparents[v] = current;
        updatePDFN(v);
        }
    }
}

BasicBlock* find(BasicBlock *v) {
    if (uf[v] != v) {
        BasicBlock *u = find(uf[v]);
        if (dfn[sdom[best[v]]] > dfn[sdom[best[uf[v]]]])
            best[v] = best[uf[v]];
        uf[v] = u;
    }
    return uf[v];
}

BasicBlock* pfind(BasicBlock *v) {
    if (puf[v] != v) {
        BasicBlock *u = pfind(puf[v]);
        if (pdfn[psdom[pbest[v]]] > pdfn[psdom[pbest[puf[v]]]])
            pbest[v] = pbest[puf[v]];
        puf[v] = u;
    }
    return puf[v];
}

void link(BasicBlock *v, BasicBlock *w) {
    uf[w] = v;
}

void plink(BasicBlock *v, BasicBlock *w) {
    puf[w] = v;
}

}

// Use the Langauer-Tarjan approach.
// https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/a%20fast%20algorithm%20for%20finding.pdf
// Loop unrolling might update dominators very frequently, and it's quite time consuming.
void Region::updateDoms() {
    updatePreds();
    // Clear existing data.
    for (auto bb : bbs) {
        bb->doms.clear();
        bb->idom = nullptr;
    }

    // Clear global data as well.
    dfn.clear();
    sdom.clear();
    vertex.clear();
    parents.clear();
    uf.clear();
    best.clear();
    num = 1;

    // For each 'u' as key, it contains all blocks that it semi-dominates.
    // 'b' for bucket.
    std::map<BasicBlock*, std::vector<BasicBlock*>> bsdom; // bsdom is the inverse of sdom.
    auto entry = getFirstBlock();
    updateDFN(entry);

    for (auto bb : bbs) {
        sdom[bb] = bb;
        uf[bb] = bb;
        best[bb] = bb;
    }

    // Deal with every block in reverse dfn order.
    for (auto it = vertex.rbegin(); it != vertex.rend(); it++) {
        auto bb = *it;
        for (auto v : bb->preds) {
        // Unreachable. Skip it.
        if (!dfn.count(v))
            continue;
        BasicBlock *u;
        if (dfn[v] < dfn[bb])
            u = v;
        else {
            find(v);
            u = best[v];
        }
        if (dfn[sdom[u]] < dfn[sdom[bb]])
            sdom[bb] = sdom[u];
    }

    bsdom[sdom[bb]].push_back(bb);
    link(parents[bb], bb);

        for (auto v : bsdom[parents[bb]]) {
            find(v);
            v->idom = sdom[best[v]] == sdom[v] ? parents[bb] : best[v];
        }
    }

    // Find idom, but ignore the entry block (which has no idom).
    for (int i = 1; i < vertex.size(); ++i) {
        auto bb = vertex[i];
        assert(bb->idom);
        if (bb->idom != sdom[bb])
        bb->idom = bb->idom->idom;
    }
    
}

void Region::updateDomFront() {
    updateDoms();
    for (auto bb : bbs)
        bb->domFront.clear();

    // Update dominance frontier.
    // See https://en.wikipedia.org/wiki/Static_single-assignment_form#Computing_minimal_SSA_using_dominance_frontiers
    // For each block, if it has at least 2 preds, then it must be at dominance frontier of all its `preds`,
    // till its `idom`.
    for (auto bb : bbs) {
        if (bb->preds.size() < 2)
        continue;

        for (auto pred : bb->preds) {
            auto runner = pred;
            while (runner != bb->idom) {
                runner->domFront.insert(bb);
                runner = runner->idom;
            }
        }
    }
}

// A dual of updateDoms().
void Region::updatePDoms() {
    updatePreds();

    std::vector<BasicBlock*> exits;
    for (auto bb : bbs) {
        if (isa<ReturnOp>(bb->getLastOp()))
        exits.push_back(bb);
    }

    if (exits.size() != 1) {
        std::cerr << "no single exit for pdom\n";
        assert(false);
    }

    auto exit = exits[0];

    for (auto bb : bbs) {
        bb->pdoms.clear();
        bb->ipdom = nullptr;
    }

    pdfn.clear();
    pvertex.clear();
    pparents.clear();
    psdom.clear();
    puf.clear();
    pbest.clear();
    pnum = 1;

    std::map<BasicBlock*, std::vector<BasicBlock*>> pbsdom;

    updatePDFN(exit);

    for (auto bb : bbs) {
        psdom[bb] = bb;
        puf[bb] = bb;
        pbest[bb] = bb;
    }

    for (auto it = pvertex.rbegin(); it != pvertex.rend(); ++it) {
        auto bb = *it;
        for (auto v : bb->succs) {
            if (!pdfn.count(v))
                continue;
            BasicBlock *u;
            if (pdfn[v] < pdfn[bb])
                u = v;
            else {
                pfind(v);
                u = pbest[v];
            }
            if (pdfn[psdom[u]] < pdfn[psdom[bb]])
                psdom[bb] = psdom[u];
        }

        pbsdom[psdom[bb]].push_back(bb);
        plink(pparents[bb], bb);

        for (auto *v : pbsdom[pparents[bb]]) {
            pfind(v);
            v->ipdom = (psdom[pbest[v]] == psdom[v]) ? pparents[bb] : pbest[v];
        }
    }

    for (size_t i = 1; i < pvertex.size(); ++i) {
        auto bb = pvertex[i];
        assert(bb->ipdom);
        if (bb->ipdom != psdom[bb])
        bb->ipdom = bb->ipdom->ipdom;
    }
}

// SSA Page 116.
// https://pfalcon.github.io/ssabook/latest/book-full.pdf
void Region::updateLiveness() {
    updatePreds();

    // Clear existing values.
    for (auto bb : bbs) {
        bb->liveIn.clear();
        bb->liveOut.clear();
    }

    std::map<BasicBlock*, std::set<Op*>> phis;
    std::map<BasicBlock*, std::set<Op*>> upwardExposed;
    std::map<BasicBlock*, std::set<Op*>> defined;

    for (auto bb : bbs) {
        for (auto op : bb->getOps()) {
            if (isa<PhiOp>(op)) {
                phis[bb].insert(op);
                continue;
            }

            defined[bb].insert(op);

            // A value is upward exposed if it's from some block upwards;
            // i.e. it's used but not defined in this block.
            for (auto value : op->getOperands()) {
                if (!defined[bb].count(value.defining))
                    upwardExposed[bb].insert(value.defining);
            }
        }
    }

    std::deque<BasicBlock*> worklist;
    
    // Do a dataflow approach. We start with all exit blocks;
    // i.e. those that have no successors.
    std::copy_if(bbs.begin(), bbs.end(), std::back_inserter(worklist), [&](BasicBlock *bb) {
        return bb->succs.size() == 0;
    });

    bool changed;
    do {
        changed = false;
        for (auto bb : bbs) {
            auto liveInOld = bb->liveIn;

            // LiveOut(B) = \bigcup_{S\in succ(B)} (LiveIn(S) - PhiDefs(S)) \cup PhiUses(B)
            // Here PhiUses(B) means the set of variables used in Phi nodes of S that come from B.
            std::set<Op*> liveOut;
            for (auto succ : bb->succs) {
                std::set_difference(
                    succ->liveIn.begin(), succ->liveIn.end(),
                    phis[succ].begin(), phis[succ].end(),
                    std::inserter(liveOut, liveOut.end())
                );
                for (auto phi : phis[succ]) {
                    auto &ops = phi->getOperands();
                    auto &attrs = phi->getAttrs();
                    for (size_t i = 0; i < ops.size(); i++) {
                        if (FROM(attrs[i]) == bb)
                        liveOut.insert(ops[i].defining);
                    }
                }
            }

            bb->liveOut = liveOut;

            // LiveIn(B) = PhiDefs(B) \cup UpwardExposed(B) \cup (LiveOut(B) - Defs(B))
            bb->liveIn.clear();
            std::set_difference(
                liveOut.begin(), liveOut.end(),
                defined[bb].begin(), defined[bb].end(),
                std::inserter(bb->liveIn, bb->liveIn.end())
            );
            for (auto x : upwardExposed[bb])
                bb->liveIn.insert(x);
            for (auto x : phis[bb])
                bb->liveIn.insert(x);

            if (liveInOld != bb->liveIn)
                changed = true;
        }
    } while (changed);

    // showLiveIn();
}

void Region::showLiveIn() {
    std::cerr << "===== live info starts =====\n";
    for (auto bb : bbs) {
        std::cerr << "=== block ===\n";
        for (auto x : bb->getOps()) {
        std::cerr << "  ";
        x->dump(std::cerr);
        }
        std::cerr << "=== livein ===\n";
        for (auto x : bb->liveIn) {
        std::cerr << "  ";
        x->dump(std::cerr);
        }
        std::cerr << "=== liveout ===\n";
        for (auto x : bb->liveOut) {
        std::cerr << "  ";
        x->dump(std::cerr);
        }
        std::cerr << "\n\n";
    }
    std::cerr << "===== live info ends =====\n\n\n";
}

void Region::dump(std::ostream &os, int depth) {
    assert(depth >= 1);
    
    os << "{\n";
    for (auto it = bbs.begin(); it != bbs.end(); it++) {
        if (bbs.size() != 1) {
        auto bb = *it;
        indent(os, depth * 2 - 2);
        os << "bb" << getBlockID(bb) << ":     // preds = [ ";
        for (auto x : bb->preds)
            os << getBlockID(x) << " ";
        
        os << "]; dom frontier = [ ";
        for (auto x : bb->getDominanceFrontier())
            os << getBlockID(x) << " ";
        
        os << "]";
        
        if (bb->idom)
            os << "; idom = " << getBlockID(bb->idom);
        os << "\n";
        }
        auto &bb = *it;
        for (auto &x : bb->getOps())
        x->dump(os, depth);
    }
    indent(os, depth * 2 - 2);
    os << "}";
}

