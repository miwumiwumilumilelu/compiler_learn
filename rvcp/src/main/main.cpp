#include "Options.h"
#include <fstream>
#include <sstream>
#include "../parse/Parser.h"
#include "../parse/Sema.h"
#include "../codegen/CodeGen.h"
#include "../opt/LowerPasses.h"
#include "../opt/passes.h"

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

  if (opts.dumpMidIR) {
    std::cerr << module;
    return 0;
  }

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

  

  std::cerr << "only --dump-mid-ir is supported in this build.\n";
  return 0;
}
