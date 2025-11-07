# 先处理代理再运行
sudo apt-get update && sudo apt-get install -y ccache
ccache -M 20G
export CCACHE_DIR=~/.cache/ccache