#include "Options.h"
#include <fstream>
#include <sstream>
#include "../parse/Parser.h"
#include "../parse/Sema.h"
#include "../codegen/CodeGen.h"
#include "../HIR-opt/HirPasses.h"
#include "../HIR-opt/AnalysisPasses.h"
#include "../HIR-opt/LoopPasses.h"
#include "../CFG-opt/LowerPasses.h"
#include "../CFG-opt/passes.h"
#include "../rv/RvPasses.h"

int main(int argc, char **argv) {
  auto opts = sys::parseArgs(argc, argv);

  std::ifstream ifs(opts.inputFile);
  if (!ifs) {
    std::cerr << "cannot open file\n";
    return 1;
  }

  std::stringstream ss;
  ss << ifs.rdbuf() << "\n";

  sys::TypeContext ctx;
  sys::Parser parser(ss.str(), ctx);
  sys::ASTNode *node = parser.parse();
  sys::Sema sema(node, ctx);

  sys::CodeGen cg(node);
  delete node;

  sys::ModuleOp *module = cg.getModule();

  sys::RaiseToFor raiseToFor(module);
  raiseToFor.run();
  auto stats = raiseToFor.stats();
  if (stats["raised-for-loops"] > 0) {
      std::cerr << "Info: Raised " << stats["raised-for-loops"] << " loops to ForOp.\n";
  }

  if (opts.dumpMidIR) {
    std::cerr << module;
    return 0;
  }

  // ForOpLower Pass
  sys::ForOpLower forOpLower(module);
  forOpLower.run();

  // MoveAlloca Pass
  sys::MoveAlloca moveAlloca(module);
  moveAlloca.run();

  // FlattenCFG Pass
  sys::FlattenCFG flatten(module);
  flatten.run();

  // Mem2Reg Pass
  auto M2R = sys::Mem2Reg(module);
  M2R.run();
  auto statistics_M2R = M2R.stats();

  if (opts.dumpCFGIR) {
    std::cerr << module;
    std::cerr << "Promoted: " << statistics_M2R["lowered-alloca"] << ", Missed: " << statistics_M2R["missed-alloca"] << "\n";
    return 0;
  }

  // Lowering Pass
  sys::rv::Lower lowerPass(module);
  lowerPass.run();
  // std::cout << "Running RISC-V Lowering..." << std::endl;
  // std::cerr << module;

  // sys::rv::Dump dumpPass(module, opts.outputFile);
  // dumpPass.run();
  sys::rv::RegAlloc regAlloc(module);
  regAlloc.run();
  
  return 0;
}
