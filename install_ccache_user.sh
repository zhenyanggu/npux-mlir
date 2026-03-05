#!/bin/bash
set -e

INSTALL_DIR="$HOME/tools/ccache"
CACHE_DIR="$HOME/.cache/ccache"
VERSION="4.12.1"

mkdir -p "$INSTALL_DIR"
mkdir -p "$CACHE_DIR"

cd "$INSTALL_DIR"

# 下载二进制 release
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    FILE="ccache-$VERSION-linux-x86_64.tar.xz"
elif [[ "$ARCH" == "aarch64" ]]; then
    FILE="ccache-$VERSION-linux-aarch64.tar.xz"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

URL="https://github.com/ccache/ccache/releases/download/v$VERSION/$FILE"
echo "Downloading $FILE ..."
wget -O "$FILE" "$URL"

# 解压
echo "Extracting..."
tar -xf "$FILE" --strip-components=1
rm "$FILE"

# 设置环境变量
export PATH="$INSTALL_DIR:$PATH"
export CCACHE_DIR="$CACHE_DIR"

# 初始化缓存
"$INSTALL_DIR/ccache" -M 20G

# 验证
echo "ccache installed successfully!"
"$INSTALL_DIR/ccache" --version
"$INSTALL_DIR/ccache" -s
