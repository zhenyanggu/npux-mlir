SHELL := /bin/bash
# 使用方式：
#   1. 列出所有可用模型: make list
#   2. 构建单个模型的所有阶段: make gelu
#   3. 构建单个阶段: make gelu.tile
#   4. 构建所有模型：make
#   5. 清理所有模型：make clean
#   6. 清理指定模型：make clean gelu
#   7. 临时跳过部分模型：make SKIP_MODELS="gelu resize"
#   8. 持久跳过某个模型：touch models/gelu/.disabled
PYTHON ?= python3
ONNX_MLIR ?= onnx-mlir
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MODELS_DIR := $(ROOT_DIR)/models
SCRIPTS_DIR := $(ROOT_DIR)/scripts
BUILD_DIR := $(ROOT_DIR)/build
CACHE_DIR := $(ROOT_DIR)/cache

# 硬件参数（单位: KB），可通过命令行覆盖
SPM_KB ?= 512
ACC_KB ?= 256
SHAPE ?=
SKIP_MODELS ?=
MODEL_PY_ARGS ?=

# 仅收集模型目录，避免把 models 下的 md 文件当作 testcase
MODEL_DIRS := $(wildcard $(MODELS_DIR)/*/)
ALL_MODEL_NAMES := $(notdir $(patsubst %/,%,$(MODEL_DIRS)))
DISABLED_MODEL_MARKERS := $(wildcard $(MODELS_DIR)/*/.disabled)
DISABLED_MODEL_NAMES_FROM_MARKERS := $(notdir $(patsubst %/,%,$(dir $(DISABLED_MODEL_MARKERS))))
DISABLED_MODEL_NAMES := $(sort $(filter $(ALL_MODEL_NAMES),$(DISABLED_MODEL_NAMES_FROM_MARKERS) $(SKIP_MODELS)))
MODEL_NAMES := $(filter-out $(DISABLED_MODEL_NAMES),$(ALL_MODEL_NAMES))
SELECTIVE_CLEAN_MODELS := $(filter $(ALL_MODEL_NAMES),$(filter-out clean,$(MAKECMDGOALS)))

.PHONY: all init run_one clean clean_cache list check_model \
	prepare tile mlir llvm zcu102 rebuild \
	$(ALL_MODEL_NAMES)

MODEL_PREPARE_TARGETS := $(addsuffix .prepare,$(ALL_MODEL_NAMES))
MODEL_TILE_TARGETS := $(addsuffix .tile,$(ALL_MODEL_NAMES))
MODEL_MLIR_TARGETS := $(addsuffix .mlir,$(ALL_MODEL_NAMES))
MODEL_LLVM_TARGETS := $(addsuffix .llvm,$(ALL_MODEL_NAMES))
MODEL_ZCU102_TARGETS := $(addsuffix .zcu102,$(ALL_MODEL_NAMES))

.PHONY: $(MODEL_PREPARE_TARGETS) $(MODEL_TILE_TARGETS) $(MODEL_MLIR_TARGETS) $(MODEL_LLVM_TARGETS) $(MODEL_ZCU102_TARGETS)

all: $(MODEL_NAMES)

init:
	@mkdir -p $(BUILD_DIR)

check_model:
	@if [[ -z "$(MODEL)" ]]; then \
		echo "[error] 请指定 MODEL，例如: make run_one MODEL=gelu"; \
		exit 2; \
	fi

run_one: check_model $(BUILD_DIR)/$(MODEL)/.zcu102.stamp

prepare: check_model $(BUILD_DIR)/$(MODEL)/.prepare.stamp

tile: check_model $(BUILD_DIR)/$(MODEL)/.tile.stamp

mlir: check_model $(BUILD_DIR)/$(MODEL)/.mlir.stamp

llvm: check_model $(BUILD_DIR)/$(MODEL)/.llvm.stamp

zcu102: check_model $(BUILD_DIR)/$(MODEL)/.zcu102.stamp

rebuild: check_model
	@rm -rf "$(BUILD_DIR)/$(MODEL)"
	@$(MAKE) run_one MODEL=$(MODEL)

ifneq (,$(filter clean,$(MAKECMDGOALS)))
$(ALL_MODEL_NAMES):
	@:
else
$(ALL_MODEL_NAMES): %: $(BUILD_DIR)/%/.zcu102.stamp
endif

$(MODEL_PREPARE_TARGETS): %.prepare: $(BUILD_DIR)/%/.prepare.stamp

$(MODEL_TILE_TARGETS): %.tile: $(BUILD_DIR)/%/.tile.stamp

$(MODEL_MLIR_TARGETS): %.mlir: $(BUILD_DIR)/%/.mlir.stamp

$(MODEL_LLVM_TARGETS): %.llvm: $(BUILD_DIR)/%/.llvm.stamp

$(MODEL_ZCU102_TARGETS): %.zcu102: $(BUILD_DIR)/%/.zcu102.stamp

$(BUILD_DIR)/%/.prepare.stamp: | init
	@model="$*"; \
	src_dir="$(MODELS_DIR)/$$model"; \
	work_dir="$(BUILD_DIR)/$$model"; \
	if [[ ! -d "$$src_dir" ]]; then \
		echo "[error] testcase '$$model' 不存在，可用: $(MODEL_NAMES)"; \
		exit 2; \
	fi; \
	mkdir -p "$$work_dir"; \
	echo "[mkdir] $$work_dir"; \
	shopt -s nullglob; \
	onnx_files=("$$src_dir"/*.onnx); \
	if (( $${#onnx_files[@]} > 0 )); then \
		cp -f "$$src_dir"/*.onnx "$$work_dir"/; \
		echo "[copy] $$src_dir/*.onnx -> $$work_dir/"; \
	fi; \
	if [[ -f "$$src_dir/model.py" ]]; then \
		cp -f "$$src_dir/model.py" "$$work_dir/model.py"; \
		echo "[gen] python $$work_dir/model.py $(MODEL_PY_ARGS)"; \
		(cd "$$work_dir" && MODEL_TEST_CACHE_DIR="$(CACHE_DIR)" $(PYTHON) model.py $(MODEL_PY_ARGS)) || { echo "[error] $$model 生成模型失败"; exit 2; }; \
	elif [[ -f "$$src_dir/model.onnx" ]]; then \
		cp -f "$$src_dir/model.onnx" "$$work_dir/model.onnx"; \
		echo "[copy] $$src_dir/model.onnx -> $$work_dir/model.onnx"; \
	else \
		echo "[error] $$model 缺少 model.py 或 model.onnx"; \
		exit 2; \
	fi; \
	if [[ ! -f "$$work_dir/model.onnx" ]]; then \
		work_onnx=("$$work_dir"/*.onnx); \
		if (( $${#work_onnx[@]} == 0 )); then \
			echo "[error] $$model 未找到任何 onnx 文件"; \
			exit 2; \
		fi; \
		cp -f "$${work_onnx[0]}" "$$work_dir/model.onnx"; \
		echo "[copy] $${work_onnx[0]} -> $$work_dir/model.onnx"; \
	fi; \
	shopt -u nullglob; \
	if [[ -f "$$src_dir/main.cpp" ]]; then \
		cp -f "$$src_dir/main.cpp" "$$work_dir/main.cpp"; \
	fi; \
	touch "$@"

$(BUILD_DIR)/%/.tile.stamp: $(BUILD_DIR)/%/.prepare.stamp
	@model="$*"; \
	work_dir="$(BUILD_DIR)/$$model"; \
	onnx_file="$$work_dir/model.onnx"; \
	tile_json="$$work_dir/tile_config.json"; \
	if [[ ! -f "$$onnx_file" ]]; then \
		echo "[error] $$model 未生成 model.onnx，无法执行分块"; \
		exit 2; \
	fi; \
	shape_args=""; \
	if [[ -n "$(SHAPE)" ]]; then \
		shape_args="--shape $(SHAPE)"; \
	elif [[ "$$model" == "mnist" ]]; then \
		shape_args="--shape 1,1,28,28"; \
		echo "[tile] $$model 未指定 SHAPE，默认使用 --shape 1,1,28,28"; \
	elif [[ "$$model" == "resnet50" ]]; then \
		shape_args="--shape 1,3,224,224"; \
		echo "[tile] $$model 未指定 SHAPE，默认使用 --shape 1,3,224,224"; \
	fi; \
	echo "[tile] $$onnx_file -> $$tile_json"; \
	$(PYTHON) "$(SCRIPTS_DIR)/conv_tile.py" \
		-i "$$onnx_file" \
		-o "$$tile_json" \
		--spm "$(SPM_KB)" \
		--acc "$(ACC_KB)" \
		$$shape_args || { echo "[error] $$model 生成 tile_config.json 失败"; exit 2; }; \
	if [[ ! -f "$$tile_json" ]]; then \
		echo "{}" > "$$tile_json"; \
		echo "[tile] $$model 无 Conv，已写入空 tile_config.json"; \
	fi; \
	touch "$@"

$(BUILD_DIR)/%/.mlir.stamp: $(BUILD_DIR)/%/.tile.stamp
	@model="$*"; \
	work_dir="$(BUILD_DIR)/$$model"; \
	onnx_file="$$work_dir/model.onnx"; \
	mlir_file="$$onnx_file.mlir"; \
	onnx_flags=""; \
	if [[ "$$model" == "resnet50" ]]; then \
		onnx_flags="--enable-conv-opt-pass=false"; \
		echo "[mlir] $$model disable ConvOptPass to keep 1x1 Conv (avoid Conv->MatMul rewrite)"; \
	fi; \
	echo "[mlir] $$onnx_file -> $$mlir_file"; \
	(cd "$$work_dir" && $(ONNX_MLIR) $$onnx_flags --EmitONNXIR "model.onnx") || { echo "[error] $$model ONNX 转 MLIR 失败"; exit 2; }; \
	if [[ ! -f "$$mlir_file" ]]; then \
		echo "[error] 未找到 MLIR 文件: $$mlir_file"; \
		exit 2; \
	fi; \
	touch "$@"

$(BUILD_DIR)/%/.llvm.stamp: $(BUILD_DIR)/%/.mlir.stamp
	@model="$*"; \
	work_dir="$(BUILD_DIR)/$$model"; \
	mlir_file="$$work_dir/model.onnx.mlir"; \
	tile_json="$$work_dir/tile_config.json"; \
	if [[ ! -f "$$tile_json" ]]; then \
		echo "{}" > "$$tile_json"; \
		echo "[tile] $$model 缺失 tile_config.json，已自动补齐空配置"; \
	fi; \
	echo "[llvm] $$mlir_file -> $$work_dir/NpuToLLVM/llvm.mlir"; \
	(cd "$$work_dir" && "$(SCRIPTS_DIR)/onnx_to_llvm.sh" "$$mlir_file" "$$tile_json") || { echo "[error] $$model MLIR Lowering 失败"; exit 2; }; \
	touch "$@"

$(BUILD_DIR)/%/.zcu102.stamp: $(BUILD_DIR)/%/.llvm.stamp
	@model="$*"; \
	work_dir="$(BUILD_DIR)/$$model"; \
	output_root="$(BUILD_DIR)/output"; \
	output_dir="$$output_root/$$model"; \
	llvm_mlir="$$work_dir/NpuToLLVM/llvm.mlir"; \
	output_bin="$$output_dir/$${model}_zcu102"; \
	compile_script="$(SCRIPTS_DIR)/compile_to_zcu102.sh"; \
	if [[ ! -f "$$compile_script" ]]; then \
		echo "[error] 缺少脚本: $$compile_script"; \
		exit 2; \
	fi; \
	if [[ ! -f "$$llvm_mlir" ]]; then \
		echo "[error] 缺少输入 MLIR: $$llvm_mlir"; \
		exit 2; \
	fi; \
	if [[ ! -f "$$work_dir/main.cpp" ]]; then \
		echo "[error] 缺少 $$work_dir/main.cpp，无法链接生成可执行文件"; \
		exit 2; \
	fi; \
	mkdir -p "$$output_dir"; \
	echo "[zcu102] $$llvm_mlir -> $$output_bin"; \
	(cd "$$work_dir" && bash "$$compile_script" "NpuToLLVM/llvm.mlir" "$$output_bin") || { echo "[error] $$model 交叉编译失败"; exit 2; }; \
	if [[ -f "$$work_dir/$${model}_input.bin" ]]; then \
		mv -f "$$work_dir/$${model}_input.bin" "$$output_dir/$${model}_input.bin"; \
	elif [[ -f "$$work_dir/input.bin" ]]; then \
		mv -f "$$work_dir/input.bin" "$$output_dir/$${model}_input.bin"; \
	fi; \
	if [[ -f "$$work_dir/$${model}_output_golden.bin" ]]; then \
		mv -f "$$work_dir/$${model}_output_golden.bin" "$$output_dir/$${model}_output_golden.bin"; \
	elif [[ -f "$$work_dir/output_golden.bin" ]]; then \
		mv -f "$$work_dir/output_golden.bin" "$$output_dir/$${model}_output_golden.bin"; \
	fi; \
	if [[ -f "$$work_dir/$${model}_labels.bin" ]]; then \
		mv -f "$$work_dir/$${model}_labels.bin" "$$output_dir/$${model}_labels.bin"; \
	elif [[ -f "$$work_dir/labels.bin" ]]; then \
		mv -f "$$work_dir/labels.bin" "$$output_dir/$${model}_labels.bin"; \
	fi; \
	if [[ -f "$$work_dir/$${model}_images_u8.bin" ]]; then \
		mv -f "$$work_dir/$${model}_images_u8.bin" "$$output_dir/$${model}_images_u8.bin"; \
	elif [[ -f "$$work_dir/images_u8.bin" ]]; then \
		mv -f "$$work_dir/images_u8.bin" "$$output_dir/$${model}_images_u8.bin"; \
	fi; \
	if [[ -f "$$work_dir/$${model}_image.pgm" ]]; then \
		mv -f "$$work_dir/$${model}_image.pgm" "$$output_dir/$${model}_image.pgm"; \
	elif [[ -f "$$work_dir/image.pgm" ]]; then \
		mv -f "$$work_dir/image.pgm" "$$output_dir/$${model}_image.pgm"; \
	fi; \
	touch "$@"

list:
	@echo "enabled models: $(MODEL_NAMES)"
	@echo "disabled models: $(DISABLED_MODEL_NAMES)"
	@echo "usage: make gelu | make gelu.tile | make run_one MODEL=gelu | make zcu102 MODEL=gelu | make tile MODEL=gelu | make llvm MODEL=gelu"
	@echo "skip temporarily: make SKIP_MODELS=\"gelu resize\""
	@echo "skip persistently: touch models/<model>/.disabled"

clean:
	@if [[ -n "$(SELECTIVE_CLEAN_MODELS)" ]]; then \
		for model in $(SELECTIVE_CLEAN_MODELS); do \
			rm -rf "$(BUILD_DIR)/$$model"; \
			rm -rf "$(BUILD_DIR)/output/$$model"; \
			echo "[clean] removed $(BUILD_DIR)/$$model"; \
			echo "[clean] removed $(BUILD_DIR)/output/$$model"; \
		done; \
	else \
		rm -rf $(BUILD_DIR); \
		echo "[clean] removed $(BUILD_DIR)"; \
	fi
	@echo "[clean] kept cache $(CACHE_DIR) (use 'make clean_cache' to remove)"

clean_cache:
	@rm -rf "$(CACHE_DIR)"
	@echo "[clean_cache] removed $(CACHE_DIR)"
