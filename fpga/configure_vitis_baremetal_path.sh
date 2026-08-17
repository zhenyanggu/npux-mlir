#!/usr/bin/env bash
set -euo pipefail

marker='# Vitis 2025.1 bare-metal cross toolchains'
path_line='export PATH=/home/lqma/Xilinx/2025.1/gnu/aarch64/lin/aarch64-none/bin:/home/lqma/Xilinx/2025.1/gnu/microblaze/lin/bin:$PATH'

if ! grep -Fqx "$marker" "$HOME/.bashrc"; then
  printf '\n%s\n%s\n' "$marker" "$path_line" >> "$HOME/.bashrc"
fi

source "$HOME/.bashrc"
command -v aarch64-none-elf-gcc
command -v mb-gcc
