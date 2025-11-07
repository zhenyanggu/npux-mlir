//===- npuxPasses.td - NPUX dialect pass definitions -----------*- tablegen -*-===//
//
// This file declares NPUX dialect passes for TableGen to generate C++ decls.
//
//===----------------------------------------------------------------------===//

#ifndef NPUX_PASSES_TD
#define NPUX_PASSES_TD

include "mlir/Pass/PassBase.td"

// Outline each NPUX op into a standalone function and replace with a call.
def OutlineNpuxOpsPass : Pass<"outline-npux-ops", "mlir::ModuleOp"> {
	let summary = "Outline each NPUX op into a func call";
	let description = [{
		For every op from the NPUX dialect, create a private func that takes the
		op's operands as arguments and returns its results, clone the op inside the
		new function, and replace the original op with a call to that function.
		No deduplication: each op instance gets its own function.
	}];
	let constructor = "createOutlineNpuxOpsPass()";
}

#endif // NPUX_PASSES_TD
