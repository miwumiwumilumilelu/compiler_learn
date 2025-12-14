#include "Options.h"
#include <fstream>
#include <sstream>
#include "../parse/Parser.h"
#include "../parse/Sema.h"
#include "../codegen/CodeGen.h"
#include "../opt/LowerPasses.h"

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

  if (opts.dumpCFGIR) {
    std::cerr << module;
    return 0;
  }

  std::cerr << "only --dump-mid-ir is supported in this build.\n";
  return 0;
}
