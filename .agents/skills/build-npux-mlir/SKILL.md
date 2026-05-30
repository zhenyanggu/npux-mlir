---
name: build-npux-mlir
description: Compile, reconfigure, and rebuild the local npux-mlir project with the repository's checked-in VS Code CMake settings. Use when Codex needs any project build in this workspace, including full builds, incremental builds, target rebuilds, or CMake cache refreshes, and when reproducing the developer's standard debug build environment.
---

# Build Npux Mlir

## Overview

Use this skill for all compilation work in this repository instead of rediscovering CMake flags by hand. Treat `.vscode/settings.json` as the source of truth for configure arguments and environment variables.

Use this skill for:

- Full builds from a fresh or existing `build/` directory.
- Incremental builds after source changes.
- Rebuilding specific generated outputs by rerunning the normal project build flow.
- Refreshing the CMake cache while keeping the repository's standard build settings.

## Build Workflow

1. Read `references/vscode-build.md` when you need the exact settings copied from `.vscode/settings.json`.
2. Run `scripts/build.sh` from the repository root to configure and build with the standard debug profile.
3. If the user asks for a clean reconfigure, delete `build/CMakeCache.txt` and rerun `scripts/build.sh`.
4. If the user asks for an incremental build, still use this same skill and rerun `scripts/build.sh` so the repository keeps one consistent build entrypoint.
5. If the user asks for a different build type or parallelism, prefer editing `scripts/build.sh` only after explicit confirmation because this skill is intended to mirror the checked-in VS Code setup.

## Commands

Run the default build with:

```bash
bash .agents/skills/build-npux-mlir/scripts/build.sh
```

The script:

- Configures `build/` from the workspace root.
- Uses the same `MLIR_DIR`, `LLVM_DIR`, `CMAKE_PREFIX_PATH`, `Python3_EXECUTABLE`, and `LLVM_ENABLE_ASSERTIONS` values as `.vscode/settings.json`.
- Prepends `~/tools/ccache` to `PATH` and exports `CCACHE_DIR=~/.cache/ccache`.
- Builds with `-j 8`.

## Execution Notes

- Keep using this skill for both clean and incremental compilation so all builds follow the same checked-in configuration.
- If the build runs in a restricted environment, writing to `~/.cache/ccache` may require elevated execution outside the sandbox.
- Do not disable `ccache` with `CCACHE_DISABLE=1` as an automatic fallback when `~/.cache/ccache` is not writable.
- If `~/.cache/ccache` is not writable, request elevated execution first or stop and report the environment limitation clearly.
- If the build runs in an unrestricted user shell, `~/.cache/ccache` should work as configured by the script.

## Notes

- Assume the LLVM and conda paths in `.vscode/settings.json` are valid unless the user says they changed.
- If configuration fails because one of those absolute paths no longer exists, report the missing path clearly and ask the user whether the skill should be updated.
- Do not invent alternative build systems for this repository when this skill is active; preserve parity with the VS Code CMake workflow.
