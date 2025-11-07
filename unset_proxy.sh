# 请使用source unset_proxy.sh来取消代理设置
# 而不是bash unset_proxy.sh，因为后者在子shell中执行，无法影响当前shell的环境变量
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
