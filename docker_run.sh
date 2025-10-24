# 1. 首先docker pull image
# docker pull xinhanguo/npu-mlir:latest

# 2. 下载并更新submodule
# git submodule update --init --recursive

# 3. 创建并启动一个带挂载的容器，并给它取个名字
docker run -d -it --name my-npux-dev \
  -v .:/workspace \
  xinhanguo/npu-mlir

# 4. 使用vscode连接到容器中,并下载cmake插件
# 5. 为了高亮，还需要安装clangd，下载插件并

# 注意，默认有代理，可以设置为自己的端口或者取消代理
# unset HTTP_PROXY
# unset HTTPS_PROXY