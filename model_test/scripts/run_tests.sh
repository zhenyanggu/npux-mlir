#!/bin/bash

# 脚本：执行所有子目录中的可执行文件并统计测试结果

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 统计变量
pass_count=0
fail_count=0
declare -a pass_tests
declare -a fail_tests

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# 依赖检查
if ! command -v sudo >/dev/null 2>&1; then
    echo -e "${RED}错误: 未找到 sudo 命令，无法按要求执行。${NC}"
    exit 2
fi

echo -e "${YELLOW}========== 开始执行测试 ==========${NC}"
echo ""

# 遍历所有子目录
for dir in */; do
    dir_name="${dir%/}"  # 移除末尾的斜杠
    
    # 跳过脚本本身
    if [ "$dir_name" = "." ]; then
        continue
    fi
    
    # 进入子目录
    cd "$dir_name" || continue
    
    echo -e "${YELLOW}正在处理目录: $dir_name${NC}"
    
    found_exec=0
    # 按命名规则查找可执行文件（例如 averagepool_zcu102）
    for file in *_zcu102; do
        [ -e "$file" ] || continue
        [ -f "$file" ] || continue
        found_exec=1

        # 添加执行权限
        if sudo chmod +x "$file"; then
            echo "  添加执行权限: $file"
        else
            echo -e "    ${RED}✗ FAIL${NC}"
            echo "    错误信息: sudo chmod +x 失败"
            ((fail_count++))
            fail_tests+=("$dir_name/$file")
            continue
        fi

        # 执行文件
        log_file="${file}.log"
        echo "  执行(sudo): $file"
        echo "  日志保存: $dir_name/$log_file"
        runtime_ok=0
        if sudo ./"$file" > "$log_file" 2>&1; then
            runtime_ok=1
        fi

        # 从日志里提取统一结果行
        # 形如: @@MODEL_TEST_RESULT@@ errors=0/768 status=PASS
        result_line="$(grep -E "@@MODEL_TEST_RESULT@@" "$log_file" | tail -n 1)"

        result_ok=-1
        err_num=""
        total_num=""
        status_str=""
        if [ -n "$result_line" ]; then
            echo "  结果行: $result_line"

            parsed_nums="$(echo "$result_line" | sed -nE 's/.*errors=([0-9]+)\/([0-9]+).*/\1 \2/p')"
            status_str="$(echo "$result_line" | sed -nE 's/.*status=([A-Za-z]+).*/\1/p' | tr '[:lower:]' '[:upper:]')"

            if [ -n "$parsed_nums" ]; then
                err_num="${parsed_nums%% *}"
                total_num="${parsed_nums##* }"
                echo "  解析结果: 错误数/总数 = $err_num/$total_num"
            fi

            if [ -n "$status_str" ]; then
                echo "  解析结果: 状态 = $status_str"
            fi

            # 优先按 status 判定；若没有 status，则回退按 errorCount 判定
            if [ "$status_str" = "PASS" ]; then
                result_ok=1
            elif [ "$status_str" = "FAIL" ]; then
                result_ok=0
            elif [ -n "$err_num" ] && [ "$err_num" -eq 0 ]; then
                result_ok=1
            elif [ -n "$err_num" ]; then
                result_ok=0
            fi
        else
            echo "  结果行: 未找到 @@MODEL_TEST_RESULT@@"
        fi

        if [ "$runtime_ok" -eq 1 ] && [ "$result_ok" -eq 1 ]; then
            echo -e "    ${GREEN}✓ PASS${NC}"
            ((pass_count++))
            pass_tests+=("$dir_name/$file")
        else
            echo -e "    ${RED}✗ FAIL${NC}"
            ((fail_count++))
            fail_tests+=("$dir_name/$file")

            if [ "$runtime_ok" -ne 1 ]; then
                echo "    失败原因: 程序执行报错（退出码非0）"
            elif [ "$result_ok" -eq 0 ]; then
                if [ -n "$status_str" ]; then
                    echo "    失败原因: status=$status_str"
                elif [ -n "$err_num" ]; then
                    echo "    失败原因: errors=$err_num/$total_num"
                else
                    echo "    失败原因: 结果判定为失败"
                fi
            else
                echo "    失败原因: 无法从日志中解析 @@MODEL_TEST_RESULT@@"
            fi

            # 显示错误信息（前10行）
            echo "    日志摘要:"
            head -10 "$log_file" | sed 's/^/      /'
        fi
    done

    if [ "$found_exec" -eq 0 ]; then
        echo "  未找到匹配的可执行文件(*_zcu102)，跳过"
    fi
    
    # 返回上级目录
    cd .. || exit 1
    echo ""
done

# 输出最终统计结果
echo -e "${YELLOW}========== 测试结果统计 ==========${NC}"
echo ""
echo -e "总计: ${GREEN}通过: $pass_count${NC} | ${RED}失败: $fail_count${NC}"
echo ""

# 显示通过的测试
if [ $pass_count -gt 0 ]; then
    echo -e "${GREEN}通过的测试:${NC}"
    for test in "${pass_tests[@]}"; do
        echo -e "  ${GREEN}✓${NC} $test"
    done
    echo ""
fi

# 显示失败的测试
if [ $fail_count -gt 0 ]; then
    echo -e "${RED}失败的测试:${NC}"
    for test in "${fail_tests[@]}"; do
        echo -e "  ${RED}✗${NC} $test"
    done
    echo ""
fi

# 返回状态码（如果有失败则返回1）
if [ $fail_count -gt 0 ]; then
    exit 1
else
    exit 0
fi
