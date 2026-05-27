#!/usr/bin/env bash

set -euo pipefail

# Return the repository root for this skill invocation.
repo_root() {
    git rev-parse --show-toplevel
}

# Build the native Ninja directory with the default parallelism from VSCode settings.
run_build() {
    local root
    root="$(repo_root)"

    cmake --build "$root/build" -- -j 8
}

# Execute the default build workflow for this repository.
main() {
    run_build
}

main "$@"
