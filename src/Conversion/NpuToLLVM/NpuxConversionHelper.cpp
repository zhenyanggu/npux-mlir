//====================================================
//src/Conversion/NpuToLLVM/NpuxConversionHelper.cpp
//this file declares helper functions for npux conversion 
//====================================================


#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"


void npux::populateLinalgToNpuxPatterns(mlir::RewritePatternSet &patterns)
{
    populateLinalgSfuToNpuxPattern(patterns);
    populateSramDataMovementPatterns(patterns);
    populateHostAllocToNpuxPatterns(patterns);
    populateNpuLifecyclePatterns(patterns);
    populateLinalgConvToNpuxPattern(patterns);
}