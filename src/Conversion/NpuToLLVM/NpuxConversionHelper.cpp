//====================================================
//src/Conversion/NpuToLLVM/NpuxConversionHelper.cpp
//this file declares helper functions for npux conversion 
//====================================================


#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"

void npux::populateNpucoreToNpuxPatterns(mlir::RewritePatternSet &patterns)
{
    populateNpucoreSfuToNpuxPattern(patterns);
    populateNpucoreComputeToNpuxPattern(patterns);
    populateSramDataMovementPatterns(patterns);
    populateHostAllocToNpuxPatterns(patterns);
    populateNpuLifecyclePatterns(patterns);
}
