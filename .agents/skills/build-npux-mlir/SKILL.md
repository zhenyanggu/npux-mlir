---
name: build-npux-mlir
description: Configure and build this npux-mlir repository with the local settings used by the current workspace. Use when Codex needs to compile this project, rerun CMake after build setting changes, rebuild targets in the default build directory, or understand which LLVM/MLIR/Python environment this repository expects from .vscode/settings.json.
---

# Build Npux Mlir

## Overview

Configure and build this repository with the same local parameters currently encoded in `.vscode/settings.json`.
Prefer the bundled scripts instead of rewriting ad-hoc `cmake` commands.

## Workflow

1. Read `references/build-config.md` to confirm the expected local toolchain, build directory, and environment variables.
2. Run `scripts/configure.sh` from the repository root to generate or refresh `build/`.
3. Run `scripts/build.sh` from the repository root to compile with the default parallelism.
4. If the user asks for cross compilation, use this skill only as the native build path and mention that `cmake_arm.sh` and `cmake_arm64.sh` are the repository's separate cross-compilation entry points.

## Native Build

Run the configure step first:

```bash
./.agents/build-npux-mlir/scripts/configure.sh
```

Then run the build step:

```bash
./.agents/build-npux-mlir/scripts/build.sh
```

The scripts assume:
- The working directory is the repository root.
- `build/` is the native build directory.
- `Ninja` is available.
- The LLVM/MLIR and Python paths from `.vscode/settings.json` are valid on this machine.

## Adjustments

If the user asks to change build type, install prefix, or another CMake option, update `scripts/configure.sh` to keep the skill as the single source of truth before rerunning it.

If the user asks why a build is failing, inspect:
- `.vscode/settings.json`
- `build/CMakeCache.txt`
- `build/CMakeFiles/CMakeOutput.log`
- `build/CMakeFiles/CMakeError.log`

## Resources

- `scripts/configure.sh`: Configure the native `build/` directory with the workspace's current VSCode CMake settings.
- `scripts/build.sh`: Build from `build/` with `-j 8`.
- `references/build-config.md`: Record the source settings and the related repository build entry points.

## Example Requests

- "Use $build-npux-mlir to configure this repo with the current VSCode CMake settings and build it."
- "Use $build-npux-mlir to rerun CMake after I changed MLIR paths."
- "Use $build-npux-mlir to explain which local Python and LLVM paths this repository is using to compile."
