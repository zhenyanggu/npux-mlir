# Generate pass declarations for npux transforms.
set(LLVM_TARGET_DEFINITIONS npuxPasses.td)
mlir_tablegen(npuxPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRnpuxTransformsIncGen)
