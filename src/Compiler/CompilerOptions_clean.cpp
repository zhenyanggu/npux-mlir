/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerOptions.cpp -------------------------===//
//
// Copyright 2022-2025 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding options.
//
//===----------------------------------------------------------------------===//
#ifndef NPUX_OPT
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Host.h"

#include "ExternalUtil.hpp"
#include "onnx-mlir/Compiler/OMCompilerRuntimeTypes.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Compiler/CompilerOptions.hpp"

#define DEBUG_TYPE "compiler_options"

// Default env var where to find default options, when defined.
const std::string OnnxMlirEnvOptionName = "ONNX_MLIR_FLAGS";

namespace onnx_mlir {

// Use external storage for the options so that they are globally accessible
std::string inputFilename;                             // common for both
std::string outputBaseName;                            // common for both
std::vector<accel::Accelerator::Kind> maccel;          // common for both
OptLevel OptimizationLevel;                            // common for both
std::string mtriple;                                   // common for both
std::string mcpu;                                      // common for both
float nnpaEpsilon;                                     // common for both
std::string march;                                     // common for both
InstrumentStages instrumentStage;                      // common for both
bool onnxConstPropRoundFPToInt;                        // common for both
int onnxConstPropExpansionBound;                       // common for both
std::vector<std::string> onnxConstPropDisablePatterns; // common for both
bool enableONNXHybridPass;                             // common for both
std::vector<std::string> functionsToDecompose;         // common for both
std::string opsForCall;                                // common for both
bool disableKrnlOpFusion;                              // common for both
bool disableQuantZeroPoint;                            // common for both
bool enableKrnlBufferReuse;                            // common for both
bool enableSafeCodeGen;                                // common for both
bool disableMemRefPrefetch;                            // common for both
uint64_t compilationNumThreads;                        // common for both
std::vector<std::string> decomposeOpsInONNX;           // common for both
EmissionTargetType emissionTarget;                     // onnx-mlir only
bool invokeOnnxVersionConverter;                       // onnx-mlir only
bool preserveLocations;                                // onnx-mlir only
bool printIR;                                          // onnx-mlir only
bool doNotEmitFullMLIRCode;                            // onnx-mlir only
bool preserveBitcode;                                  // onnx-mlir only
bool preserveLLVMIR;                                   // onnx-mlir only
bool preserveMLIR;                                     // onnx-mlir only
bool useOnnxModelTypes;                                // onnx-mlir only
int repeatOnnxTransform;                               // onnx-mlir only
std::string shapeInformation;                          // onnx-mlir only
std::string dimParams;                                 // onnx-mlir only
ModelSize modelSize;                                   // onnx-mlir only
bool storeConstantsToFile;                             // onnx-mlir only
float constantsToFileTotalThreshold;                   // onnx-mlir only
float constantsToFileSingleThreshold;                  // onnx-mlir only
bool VerboseOutput;                                    // onnx-mlir only
std::vector<std::string> Xopt;                         // onnx-mlir only
std::vector<std::string> Xllc;                         // onnx-mlir only
std::string mllvm;                                     // onnx-mlir only
std::string instrumentOps;                             // onnx-mlir only
unsigned instrumentControlBits;                        // onnx-mlir only
std::string parallelizeOps;                            // onnx-mlir only
std::string instrumentSignatures;                      // onnx-mlir only
std::string instrumentOnnxNode;                        // onnx-mlir only
std::string ONNXOpStats;                               // onnx-mlir only
int onnxOpTransformThreshold;                          // onnx-mlir only
bool onnxOpTransformReport;                            // onnx-mlir only
bool enableParallel;                                   // onnx-mlir only
bool disableSimdOption;                                // onnx-mlir only
bool enableFastMathOption;                             // onnx-mlir only
bool disableRecomposeOption;                           // onnx-mlir only
bool enableSimdDataLayout;                             // onnx-mlir only
bool verifyInputTensors;                               // onnx-mlir only
bool allowSorting;                                     // onnx-mlir only
std::vector<std::string> reportHeapBefore;             // onnx-mlir only
std::vector<std::string> reportHeapAfter;              // onnx-mlir only
std::string modelTag;                                  // onnx-mlir only
bool enableConvOptPass;                                // onnx-mlir only
std::vector<std::string> replaceOpWithItsOperand;      // onnx-mlir only
bool disableConstantProp;                              // onnx-mlir only
std::vector<std::string> extraLibPaths;                // onnx-mlir only
std::vector<std::string> extraLibs;                    // onnx-mlir only
ProfileIRs profileIR;                                  // onnx-mlir only
OptReport optReport;                                   // onnx-mlir only
bool enableTiming;                                     // onnx-mlir only
bool enableBoundCheck;                                 // onnx-mlir only
bool split_input_file;                                 // onnx-mlir-opt only
bool verify_diagnostics;                               // onnx-mlir-opt only
bool verify_passes;                                    // onnx-mlir-opt only
bool allowUnregisteredDialects;                        // onnx-mlir-opt only


// Configuration states associated with certain options.
// For example, when maccel is specified, NNPA can register
// dependent libdnn.
// This is just a simple string to vector map currently.
// If it gets more complicated in the future, it can be
// replaced by a class of its own.
std::map<std::string, std::vector<std::string>> CompilerConfigMap;
std::map<std::string, std::vector<size_t>> CompilerConfigStack;

// Must match ModelSize enum
const std::string modelSizeStr[] = {"small", "medium", "large", "huge"};

std::string customEnvFlags;

// =============================================================================
// Methods for setting and getting compiler variables.

// The customEnvFlags must be scanned before the normal options.
bool parseCustomEnvFlagsCommandLineOption(
    int argc, const char *const *argv, llvm::raw_ostream *errs) {
  // Use the default ONNX MLIR Environment variable, unless specified otherwise
  // by an argument, see below.
  std::string envVar = OnnxMlirEnvOptionName;
  // VerboseOutput is not yet set, so scan ourselves.
  bool verbose = false;
  // Customized version? -customEnvFlags=val and save its value.
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.find("--customEnvFlags") == 0) {
      envVar = arg.substr(sizeof("--customEnvFlags"));
    } else if (arg.find("-customEnvFlags") == 0) {
      envVar = arg.substr(sizeof("-customEnvFlags"));
    } else if (arg.compare("-v") == 0) {
      verbose = true;
    }
  }
  // Check that the env var does not recursively hold another -customEnvFlags.
  const char *envValCstr;
  if ((envValCstr = std::getenv(envVar.c_str()))) {
    std::string envVal(envValCstr);
    if (envVal.find("-customEnvFlags") != std::string::npos) {
      if (errs)
        *errs << "\nWarning: recursive use of --customEnvFlags in "
                 "environment flag not permited\n\n";
      return false;
    }
    if (envVal.find("-v") != std::string::npos)
      verbose = true;
    if (verbose)
      printf("Onnx-mlir default options from '%s' are '%s'.\n", envVar.c_str(),
          envValCstr);
  }
  if (verbose && argc > 0) {
    printf("Onnx-mlir command: %s", argv[0]);
    if (envValCstr)
      printf(" %s", envValCstr);
    for (int i = 1; i < argc; ++i)
      printf(" %s", argv[i]);
    printf("\n");
  }
  // The envVar is verified, use it.
  setCustomEnvVar(envVar);
  return true;
}

// Support for customEnvFlags.
void setCustomEnvVar(const std::string &envVarName) {
  assert(envVarName != "" && "Expecting valid target envVarName description");
  LLVM_DEBUG(
      llvm::dbgs() << DEBUG_TYPE << "Set envVarName\"" << envVarName << "\"\n");
  customEnvFlags = envVarName;
}

void clearCustomEnvVar() { customEnvFlags.clear(); }

std::string getCustomEnvVarOption() {
  return (customEnvFlags != "") ? "--customEnvFlags=" + customEnvFlags : "";
}

// Support for Triple.
void setTargetTriple(const std::string &triple) {
  assert(triple != "" && "Expecting valid target triple description");
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set triple\"" << triple << "\"\n");
  mtriple = triple;
}

void clearTargetTriple() { mtriple.clear(); }

std::string getTargetTripleOption() {
  std::string targetOptions = "";
  // Command cannot tolerate extra spaces. Add only when needed.
  if (mtriple != "")
    targetOptions = "--mtriple=" + mtriple;
  else if (kDefaultTriple != "")
    targetOptions = "--mtriple=" + kDefaultTriple;
  return targetOptions;
}

// Support for Arch.
void setTargetArch(const std::string &arch) {
  assert(arch != "" && "Expecting valid target arch description");
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set arch\"" << arch << "\"\n");
  march = arch;
}

void clearTargetArch() { march.clear(); }

// Sort out architectures for Z systems (hybrid archXX and zYY names).
static int64_t decodeZArchNum(std::string str) {
  if (str == "arch12" || str == "z14") // Z14 and equivalents.
    return 12;
  if (str == "arch13" || str == "z15") // Z15 and equivalents.
    return 13;
  if (str == "arch14" || str == "z16") // Z16 and equivalents.
    return 14;
  if (str == "arch15" || str == "z17") // Z17 and equivalents.
    return 15;
  return -1;
}

int64_t getZArchNum(const std::string &arch, const std::string cpu) {
  // Give priority to march, use (deprecated) mcpu if march is not defined.
  int64_t num = decodeZArchNum(arch);
  if (num == -1)
    num = decodeZArchNum(cpu);
  return num;
}

std::string getTargetArchOption(bool forLLVMToolchain) {
  // LLVM toolchain wants a --march=systemz for all z machines; the specific
  // Z architecture will be specified with the LLVM Toolchain --mcpu.
  if (forLLVMToolchain) {
    // Handle special case for Z.
    int64_t zArchNum = getZArchNum(march, mcpu);
    if (zArchNum != -1)
      return "--march=systemz";
    // On mac, llc --version seem to want aarch64 or arm64.
    if (march == "apple-m1" || march == "apple-m2" || march == "apple-m3" ||
        march == "apple-m4")
      return "--march=arm64";
  }
  return (march != "") ? "--march=" + march : "";
}

// Support for CPU.
void setTargetCPU(const std::string &cpu) {
  assert(cpu != "" && "Expecting valid target cpu description");
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set CPU\"" << cpu << "\"\n");
  mcpu = cpu;
}

void clearTargetCPU() { mcpu.clear(); }

// As the LLVM tooling for Z may not support the latest, cap it by this
// --mcpu=arch{MAX_LLVM_Z_ARCH_LEVEL} value.
#define MAX_LLVM_Z_ARCH_LEVEL 14

std::string getTargetCPUOption(bool forLLVMToolchain, bool cpuOnly) {
  // With cpu only, return the mcpu value; without it, prepend with "--mcpu=".
  std::string str = (cpuOnly ? "" : "--mcpu=");

  // The LLVM toolchain wants the specific Z architecture to be expressed with
  // the LLVM Toolchain --mcpu. Convert below the --march into their
  // corresponding --mcpu equivalent.
  if (forLLVMToolchain) {
    // Handle special case for Z.
    int64_t zArchNum = getZArchNum(march, mcpu);
    if (zArchNum != -1) {
      // Cap at max supported LLVM level.
      zArchNum = std::min(zArchNum, (int64_t)MAX_LLVM_Z_ARCH_LEVEL);
      return str.append("arch" + std::to_string(zArchNum));
    }
  }
  return (mcpu != "") ? str + mcpu : "";
}

// Support for Accel.
static bool getAccelKindFromString(
    accel::Accelerator::Kind &kind, const std::string &str) {
  // Test each existing accelerator, returning its Kind when found.
  APPLY_TO_ACCELERATORS(ACCEL_CL_ENUM_FROM_STRING, kind, str);
  // No specific accelerator found, check if we have Kind::NONE
  kind = accel::Accelerator::Kind::NONE;
  return str.compare(std::string("NONE")) == 0;
}

// Return 0 on success, nonzero on error.
int setTargetAccel(const std::string &str) {
  assert(str != "" && "Expecting valid accelerator description");
  accel::Accelerator::Kind accelKind;
  if (getAccelKindFromString(accelKind, str)) {
    setTargetAccel(accelKind);
    return 0;
  }
  return 1;
}

void setTargetAccel(const accel::Accelerator::Kind accel) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set accel\"" << accel << "\"\n";);
  // Add accel to maccel.
  maccel.push_back(accel);
}

void clearTargetAccel() {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Clearing accel\n");
  maccel.clear();
}

std::string getTargetAccel() {
  std::stringstream ss;
  int accelCount = 0;
  for (accel::Accelerator::Kind accel : maccel) {
    if (accelCount++)
      ss << " ";
    ss << "--maccel=" << accel;
  }
  if (!accelCount)
    ss << "--maccel=NONE";
  return ss.str();
}

// Support for Optimization level.
void setOptLevel(const OptLevel level) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set opt level " << level << "\n");
  OptimizationLevel = level;
}

void clearOptLevel() { OptimizationLevel = OptLevel::O0; }

std::string getOptimizationLevelOption() {
  switch (OptimizationLevel) {
  case OptLevel::O0:
    return "-O0";
  case OptLevel::O1:
    return "-O1";
  case OptLevel::O2:
    return "-O2";
  case OptLevel::O3:
    return "-O3";
  }
  llvm_unreachable("Unexpected optimization level");
  return "";
}

// Support for Xopt.
void setXoptOption(const std::vector<std::string> &flags) {
  for (const std::string &flag : flags)
    Xopt.push_back(flag);
}

void clearXoptOption() { Xopt.clear(); }

std::vector<std::string> getXoptOption() {
  if (Xopt.empty())
    return std::vector<std::string>();

  std::vector<std::string> flags;
  for (std::string flag : Xopt)
    flags.push_back(flag);

  return flags;
}

// Support for Xllc.
void setXllcOption(const std::vector<std::string> &flags) {
  for (const std::string &flag : flags)
    Xllc.push_back(flag);
}

void clearXllcOption() { Xllc.clear(); }

std::vector<std::string> getXllcOption() {
  if (Xllc.empty())
    return std::vector<std::string>();

  std::vector<std::string> flags;
  for (std::string flag : Xllc)
    flags.push_back(flag);

  return flags;
}

// Support for LLVM.
void setLLVMOption(const std::string &flag) { mllvm = flag; }
void clearLLVMOption() { mllvm.clear(); }
std::string getLLVMOption() { return (mllvm != "") ? mllvm : std::string(); }

static std::vector<std::string> split(std::string &input) {
  std::stringstream ss(input);
  std::istream_iterator<std::string> begin(ss);
  std::istream_iterator<std::string> end;
  std::vector<std::string> vstrings(begin, end);
  return vstrings;
}

std::vector<std::string> getLLVMOptions() {
  if (mllvm == "")
    return std::vector<std::string>();

  return split(mllvm);
}

// Support for model tag
void setModelTag(const std::string &str) { modelTag = str; }
void clearModelTag() { modelTag = ""; }
std::string getModelTag() { return modelTag; }

// Support for Verbose Option
void setVerboseOption() { VerboseOutput = true; }
void clearVerboseOption() { VerboseOutput = false; }
std::string getVerboseOption() {
  return VerboseOutput ? std::string("-v") : std::string();
}

// =============================================================================
// Methods for OMCompilerOptions

int setCompilerOption(const OptionKind kind, const std::string &val) {
  switch (kind) {
  case OptionKind::TargetTriple:
    setTargetTriple(val);
    break;
  case OptionKind::TargetArch:
    setTargetArch(val);
    break;
  case OptionKind::TargetCPU:
    setTargetCPU(val);
    break;
  case OptionKind::TargetAccel:
    if (setTargetAccel(val) != 0)
      return InvalidCompilerOption;
    break;
  case OptionKind::CompilerOptLevel: {
    int level = atoi(val.c_str());
    if (level < 0 || level > 3)
      return InvalidCompilerOption;
    setOptLevel((OptLevel)level);
  } break;
  case OptionKind::OPTFlag:
    setXoptOption({val});
    break;
  case OptionKind::LLCFlag:
    setXllcOption({val});
    break;
  case OptionKind::LLVMFlag:
    setLLVMOption(val);
    break;
  case OptionKind::ModelTag:
    setModelTag(val);
    break;
  case OptionKind::Verbose:
    setVerboseOption();
    break;
    // Ignore options that were added but are unknown.
  }
  return CompilerSuccess;
}

void clearCompilerOption(const OptionKind kind) {
  switch (kind) {
  case OptionKind::TargetTriple:
    clearTargetTriple();
    break;
  case OptionKind::TargetArch:
    clearTargetArch();
    break;
  case OptionKind::TargetCPU:
    clearTargetCPU();
    break;
  case OptionKind::TargetAccel:
    clearTargetAccel();
    break;
  case OptionKind::CompilerOptLevel:
    clearOptLevel();
    break;
  case OptionKind::OPTFlag:
    clearXoptOption();
    break;
  case OptionKind::LLCFlag:
    clearXllcOption();
    break;
  case OptionKind::LLVMFlag:
    clearLLVMOption();
    break;
  case OptionKind::ModelTag:
    clearModelTag();
    break;
  case OptionKind::Verbose:
    clearVerboseOption();
    break;
    // Ignore options that were added but are unknown.
  }
}

std::string getCompilerOption(const OptionKind kind) {
  switch (kind) {
  case OptionKind::TargetTriple:
    return getTargetTripleOption();
  case OptionKind::TargetArch:
    return getTargetArchOption();
  case OptionKind::TargetCPU:
    return getTargetCPUOption();
  case OptionKind::TargetAccel:
    return getTargetAccel();
  case OptionKind::CompilerOptLevel:
    return getOptimizationLevelOption();
  case OptionKind::OPTFlag:
  case OptionKind::LLCFlag: {
    std::vector<std::string> flags =
        (kind == OptionKind::OPTFlag) ? getXoptOption() : getXllcOption();
    std::stringstream ss;
    for (int i = 0, n = flags.size(); i < n; ++i) {
      ss << flags.at(i);
      if (i != n - 1)
        ss << ' ';
    }
    return ss.str();
  }
  case OptionKind::LLVMFlag:
    return getLLVMOption();
  case OptionKind::ModelTag:
    return getModelTag();
  case OptionKind::Verbose:
    return getVerboseOption();
  }
  return std::string();
}

int setCompilerOptions(const CompilerOptionList &list) {
  for (const auto &pair : list) {
    int rc = setCompilerOption(pair.first, pair.second);
    if (rc != CompilerSuccess)
      return rc;
  }
  return CompilerSuccess;
}

// Get the string vector associated with the specified key
std::vector<std::string> getCompilerConfig(std::string k) {
  return CompilerConfigMap[k];
}

// Add strings in a vector to the string vector associated
// with the specified key
void addCompilerConfig(std::string k, std::vector<std::string> v, bool head) {
  std::vector<std::string> u = CompilerConfigMap[k];

  u.insert(head ? u.begin() : u.end(), v.begin(), v.end());
  CompilerConfigMap[k] = u;
}

// Delete strings in a vector from the string vector associated
// with the specified key
void delCompilerConfig(std::string k, std::vector<std::string> v) {
  std::vector<std::string> u = CompilerConfigMap[k];

  u.erase(remove_if(begin(u), end(u),
              [&](auto x) { return find(begin(v), end(v), x) != end(v); }),
      end(u));
  CompilerConfigMap[k] = u;
}

std::optional<std::string> getEnvVar(std::string name) {
  if (const char *envVar = std::getenv(name.c_str()))
    return std::string(envVar);
  return std::nullopt;
}

// Find the path to the onnx-mlir executable
std::string getExecPath() {
  // argv0 is only used as a fallback for rare environments
  // where /proc isn't mounted and mainExecAddr is only needed for
  // unknown unix-like platforms
  auto execPath = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  if (execPath.empty()) {
    llvm::errs()
        << "\nWarning: Could not find path to current executable, falling "
           "back to default install path: "
        << kExecPath << "\n\n";
    return kExecPath;
  }
  return execPath;
}

// Directory contains all the libraries, jars, etc. that are necessary for
// running onnx-mlir. It's resolved in the following order:
//
//   - if ONNX_MLIR_LIBRARY_PATH is set, use it, otherwise
//   - get path from where onnx-mlir is run, if it's of the form
//     /foo/bar/bin/onnx-mlir,
//     the runtime directory is /foo/bar/lib (note that when onnx-mlir is
//     installed system wide, which is typically /usr/local/bin, this will
//     correctly resolve to /usr/local/lib), but some systems still have
//     lib64 so we check that first. If neither exists, then
//   - use CMAKE_INSTALL_PREFIX/lib, which is typically /usr/local/lib
//
// We now explicitly set CMAKE_INSTALL_LIBDIR to lib so we don't have
// to deal with lib64 anymore.
std::string getLibraryPath() {
  const auto &envDir = getEnvVar("ONNX_MLIR_LIBRARY_PATH");
  if (envDir && llvm::sys::fs::exists(envDir.value()))
    return envDir.value();

  std::string execDir = llvm::sys::path::parent_path(getExecPath()).str();
  if (llvm::sys::path::stem(execDir).str().compare("bin") == 0) {
    std::string p = execDir.substr(0, execDir.size() - 3);
    if (llvm::sys::fs::exists(p + "lib"))
      return p + "lib";
  }

  llvm::SmallString<8> instDir(kInstPath);
  llvm::sys::path::append(instDir, "lib");
  return llvm::StringRef(instDir).str();
}

// onnx-mlir currently requires llvm tools llc and opt and they are assumed
// to be under llvm-project/build/bin. This doesn't work with the case where
// llvm-project has been installed system wide (typically under
// /usr/local/...) and its source has been removed.
//
// To account for this scenario, we first search for the tools in the same
// directory where onnx-mlir is run. If they are found, it means both
// onnx-mlir and llvm-project have been installed system wide under the same
// directory, so we get them from that directory (typically /usr/local/bin).
// Otherwise, at least one of onnx-mlir and llvm-project has not been
// installed system wide. In this case, getToolPath returns the fallback
// directory where llvm is built which is typically llvm-project/build/bin.
//
// Note that this will not work if both onnx-mlir and llvm-project have been
// installed system wide but to different places and their sources have been
// removed. So we force CMAKE_INSTALL_PREFIX to be the same as that of
// llvm-project.
//
// If the flag is true, getToolPath will simply return the path detected by
// cmake at compile time. This is used for system wide tools such as cc, ld,
// ar, etc. Note that this means the path is valid only on the system where
// onnx-mlir is built. If onnx-mlir is subsequently run on a system that does
// not have these tools installed in the "standard" places, it will fail.
//
// Setting flag = true is also used to simply look up non-path config such
// as lrodataScript.
std::string getToolPath(
    const std::string &tool, bool flag /*false by default*/) {
  if (!flag) {
    std::string execDir = llvm::sys::path::parent_path(getExecPath()).str();
    llvm::SmallString<8> toolPath(execDir);
    llvm::sys::path::append(toolPath, tool);
    std::string p = llvm::StringRef(toolPath).str();
    if (llvm::sys::fs::can_execute(p))
      return p;
  }

  return toolPathMap.at(tool);
}

// This function is called before llvm::cl::ParseCommandLineOptions
// to remove unrelated options in addition to hiding them. Since
// hiding only means that unrelated options will not be printed by
// -h|--help but they can still be used and silently ignored. But
// the correct behavior is that using a unrelated option should
// result in a unknown option error.
void removeUnrelatedOptions(
    const std::vector<llvm::cl::OptionCategory *> Categories) {
  // Do not remove LLVM "internal" options such as --debug
  // that do not have a category (and therefore placed
  // under the general category). So we add the general
  // category to the list of not-really-hidden options.
  std::vector<llvm::cl::OptionCategory *> optCategories(Categories);
  optCategories.push_back(&llvm::cl::getGeneralCategory());
  llvm::cl::HideUnrelatedOptions(optCategories);

  llvm::StringMap<llvm::cl::Option *> &optMap =
      llvm::cl::getRegisteredOptions();
  for (auto n = optMap.begin(); n != optMap.end(); n++) {
    llvm::cl::Option *opt = n->getValue();
    if (opt->getOptionHiddenFlag() == llvm::cl::ReallyHidden)
      opt->removeArgument();
  }
}

// This function can be called after llvm::cl::ParseCommandLineOptions
// to create whatever options related compiler configuration states
// based on the parsed options. It can also check for option consistency.
//
// The reason we don't put llvm::cl::ParseCommandLineOptions and
// initCompilerConfig in a single function is that according to llvm doc
// llvm::cl::ParseCommandLineOptions should be called from main.
void initCompilerConfig() {
  // Test option requirements.
  if (!ONNXOpStats.empty() && emissionTarget <= EmitONNXIR)
    llvm::errs()
        << "\nWarning: --onnx-op-stats requires targets like --EmitMLIR, "
           "--EmitLLVMIR, or binary-generating emit commands.\n\n";

  // Library setup for EmitLib and EmitJNI targets
  if (emissionTarget == EmitLib || emissionTarget == EmitJNI) {
    // Add mandatory libs
    addCompilerConfig(CCM_SHARED_LIB_DEPS,
        emissionTarget == EmitLib
            ? std::vector<std::string>{"cruntime"}
            : std::vector<std::string>{"jniruntime", "cruntime"});
    addCompilerConfig(CCM_SHARED_LIB_PATH_DEPS, {getLibraryPath()});

    // Add OpenMP LLVM library if parallel is enabled.
    if (enableParallel)
      addCompilerConfig(CCM_SHARED_LIB_DEPS, {"ompruntime"});

    // Add user specified libs and their path
    // Multiple lib or directory can be specified with multiple options.
    // For example, -lextra1, -lextra2, -Lpath1, -Lpath2
    addCompilerConfig(CCM_SHARED_LIB_DEPS, extraLibs);
    addCompilerConfig(CCM_SHARED_LIB_PATH_DEPS, extraLibPaths);
  }

  // Enable aggressive optimization for NNPA with -O3
  if (OptimizationLevel == OptLevel::O3 &&
      getTargetAccel().find("NNPA") != std::string::npos) {
    // Have O3 and NNPA. May enable fast math default in the future.
  }

  // Enabling unsafe math.
  if (enableFastMathOption &&
      getLLVMOption().find("enable-unsafe-fp-math") == std::string::npos) {
    // Fast math option is enabled (in general)
    setLLVMOption(getLLVMOption() + " --enable-unsafe-fp-math");
  }

  if (march == "native") {
    march = std::string(llvm::sys::getHostCPUName());
    if (VerboseOutput)
      llvm::outs() << "Native machine set as \"" << march << "\"\n";
  }
}

bool hasInstrumentation(InstrumentStages targetInstrumentationStage) {
  // Want it here?
  if (instrumentStage != targetInstrumentationStage)
    return false;
  // Now check if we are time/memory instrumenting anything.
  return (instrumentOps != "" && instrumentOps != "NONE");
}

bool hasSignatureInstrumentation(InstrumentStages targetInstrumentationStage) {
  // Want it here?
  if (instrumentStage != targetInstrumentationStage)
    return false;
  // Now check if we are signature instrumenting anything.
  return (instrumentSignatures != "" && instrumentSignatures != "NONE") ||
         (instrumentOnnxNode != "" && instrumentOnnxNode != "NONE");
}

} // namespace onnx_mlir
#endif