//===- npuxPasses.td - NPUX dialect pass definitions -----------*- tablegen -*-===//
//
// This file declares NPUX dialect passes for TableGen to generate C++ decls.
//
//===----------------------------------------------------------------------===//

#ifndef NPUX_PASSES_TD
#define NPUX_PASSES_TD

include "mlir/Pass/PassBase.td"

// Minimal example pass to seed generation; extend/rename as needed.
def NPUXExamplePass : Pass<"npux-example", "mlir::func::FuncOp"> {
	let summary = "Example NPUX pass placeholder";
	let description = [{
		Placeholder pass for NPUX dialect. Replace with real passes and factories.
	}];
	// Optional: when you implement it, expose a factory function with this name.
	// let constructor = "createNPUXExamplePass()";
}

#endif // NPUX_PASSES_TD
