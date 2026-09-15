# ── RWKV-7 CPU Inference Library — Build ──────────────────────────
#
# A SINGLE build supports BOTH fp16 ("RWKV7") and bf16 ("RWKV7B")
# weight binaries: src/rwkv_api.cpp compiles the engine twice
# (namespace Fp16 + namespace Bf16) and dispatches at runtime.
#
# Targets:
#   make all          — build shared library + all tests
#   make lib          — build librwkv.so (handles fp16 AND bf16)
#   make lib-bf16     — alias → librwkv_bf16.so (same dual-format binary)
#   make test_speed   — build speed benchmark
#   make test_correct — build correctness test
#   make test_utils16 — build 16-bit conversion unit tests (fp16 + bf16)
#   make test_ablate  — build with all scalar kernels (debug baseline)
#   make clean        — remove build artifacts

CXX      ?= g++
CXX_ABLATE = $(CXX_FAST) -DABLATE_LAYER_NORM -DABLATE_GROUP_NORM -DABLATE_TIME_SHIFT -DABLATE_VEC_ADD -DABLATE_VEC_MUL -DABLATE_RELU_SQ -DABLATE_K_NORMALIZE -DABLATE_WKV -DABLATE_GEMV -DABLATE_SPARSE_FFN
CXX_PIC   = -O3 -march=native -fopenmp -fPIC -flto -DRWKV_BUILD_DLL -Wno-unused-result
CXX_BIND  = -DRWKV_BIND_CORES=4
# CXX_BIND += -DRWKV_SPIN_WAIT   # uncomment to enable OMP spin-wait
CXX_FAST  = -O3 -march=native -fopenmp -flto -funroll-loops -falign-loops=64 -Wno-unused-result $(CXX_BIND)
# CXX_PGO_G = $(CXX_FAST) -fprofile-generate -fprofile-dir=build/pgo
# CXX_PGO_U = $(CXX_FAST) -fprofile-use -fprofile-correction -fprofile-dir=build/pgo
LDFLAGS   = -lm -flto
INCLUDES  = -Isrc -Isrc/npu -Isrc/sampling -Ithird_party/openvino/include

# OpenVINO library path (from pip wheel)
OVLIB     := $(shell python3 -c 'import openvino,os;print(os.path.join(os.path.dirname(openvino.__file__),"libs"))')
OV_LDFLAGS = -L$(OVLIB) -l:libopenvino.so.2631 -Wl,-rpath,$(OVLIB)

SRC_DIR   = src
TEST_DIR  = tests
BUILD_DIR = build

# ── Library sources ───────────────────────────────────────────────
# The dispatcher TU compiles the whole engine twice (fp16 + bf16).
LIB_SRCS  = $(SRC_DIR)/rwkv_api.cpp

# engine files pulled in by rwkv_api.cpp (rebuild triggers)
ENGINE_DEPS = $(SRC_DIR)/rwkv_lib.cpp $(SRC_DIR)/gemm_fp16.cpp \
              $(SRC_DIR)/rwkv.h $(SRC_DIR)/gemv_f16.h $(SRC_DIR)/gemm_fp16.h \
              $(SRC_DIR)/utils_16.h $(SRC_DIR)/alloc.h

LIB_NAME      = librwkv.so
LIB_NAME_BF16 = librwkv_bf16.so

# ── Default target ────────────────────────────────────────────────
.PHONY: all
all: lib test_speed test_correct test_spec test_utils16 test_utils16_bf16 chat

# ── Build directory ───────────────────────────────────────────────
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Shared library (.so with -fPIC) ───────────────────────────────
.PHONY: lib
lib: $(LIB_NAME)

.PHONY: lib-bf16
lib-bf16: $(LIB_NAME_BF16)

$(LIB_NAME): $(LIB_SRCS) $(ENGINE_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXX_PIC) $(INCLUDES) -shared -o $@ $(LIB_SRCS) $(LDFLAGS)

# bf16 artifact: same dual-format binary
$(LIB_NAME_BF16): $(LIB_NAME)
	cp $(LIB_NAME) $@

# ── Tests (compiled without -fPIC for max speed) ──────────────────
$(BUILD_DIR)/test_speed: $(TEST_DIR)/test_speed.cpp $(LIB_SRCS) $(ENGINE_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXX_FAST) $(INCLUDES) -o $@ $< $(LIB_SRCS) $(LDFLAGS)

$(BUILD_DIR)/test_correctness: $(TEST_DIR)/test_correctness.cpp $(LIB_SRCS) $(ENGINE_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXX_FAST) $(INCLUDES) -o $@ $< $(LIB_SRCS) $(LDFLAGS)

$(BUILD_DIR)/test_spec: $(TEST_DIR)/test_spec.cpp $(LIB_SRCS) $(ENGINE_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXX_FAST) $(INCLUDES) -o $@ $< $(LIB_SRCS) $(LDFLAGS)

$(BUILD_DIR)/test_ablate: $(TEST_DIR)/test_ablate.cpp $(LIB_SRCS) $(ENGINE_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXX_ABLATE) $(INCLUDES) -o $@ $< $(LIB_SRCS) $(LDFLAGS)

$(BUILD_DIR)/test_utils16: $(TEST_DIR)/test_utils16.cpp $(SRC_DIR)/utils_16.cpp $(SRC_DIR)/utils_16.h | $(BUILD_DIR)
	$(CXX) $(CXX_FAST) $(INCLUDES) -o $@ $< $(SRC_DIR)/utils_16.cpp $(LDFLAGS)

# bf16 unit tests: same test, default format = bf16 (exercises bf16 path)
$(BUILD_DIR)/test_utils16_bf16: $(TEST_DIR)/test_utils16.cpp $(SRC_DIR)/utils_16.cpp $(SRC_DIR)/utils_16.h | $(BUILD_DIR)
	$(CXX) $(CXX_FAST) $(INCLUDES) -DRWKV_WEIGHTS_BF16 -o $@ $< $(SRC_DIR)/utils_16.cpp $(LDFLAGS)

# bf16 engine binaries: aliases — the combined engine already
# handles both formats, so they are copies of the base binaries
$(BUILD_DIR)/test_correctness_bf16: $(BUILD_DIR)/test_correctness
	cp $< $@

$(BUILD_DIR)/test_speed_bf16: $(BUILD_DIR)/test_speed
	cp $< $@

.PHONY: test_speed
test_speed: $(BUILD_DIR)/test_speed

.PHONY: test_correct
test_correct: $(BUILD_DIR)/test_correctness

.PHONY: test_spec
test_spec: $(BUILD_DIR)/test_spec

.PHONY: test_utils16
test_utils16: $(BUILD_DIR)/test_utils16

.PHONY: test_utils16_bf16
test_utils16_bf16: $(BUILD_DIR)/test_utils16_bf16

.PHONY: test_correct_bf16
test_correct_bf16: $(BUILD_DIR)/test_correctness_bf16

.PHONY: test_speed_bf16
test_speed_bf16: $(BUILD_DIR)/test_speed_bf16

.PHONY: test_ablate
test_ablate: $(BUILD_DIR)/test_ablate

# # ── PGO (Profile-Guided Optimization) ───────────────────────────
# .PHONY: pgo
# pgo: | $(BUILD_DIR)
# 	mkdir -p $(BUILD_DIR)/pgo
# 	$(CXX) $(CXX_PGO_G) $(INCLUDES) -o $(BUILD_DIR)/test_speed_pgo $(TEST_DIR)/test_speed.cpp $(LIB_SRCS) $(LDFLAGS)
# 	$(BUILD_DIR)/test_speed_pgo rwkv7_0.1b_fp16.bin 200 10
# 	$(CXX) $(CXX_PGO_U) $(INCLUDES) -o $(BUILD_DIR)/test_speed_pgo $(TEST_DIR)/test_speed.cpp $(LIB_SRCS) $(LDFLAGS)
# 	rm -f $(BUILD_DIR)/*.gcda
# 	rm -rf $(BUILD_DIR)/pgo
# 	@echo "PGO optimized binary: build/test_speed_pgo"

# ── Chat example (with NPU speculative decoding) ──────────────────
SPEC_SRCS = $(SRC_DIR)/npu/spec.cpp
SPEC_DEPS = $(SRC_DIR)/npu/spec.cpp $(SRC_DIR)/npu/spec_api.h $(SRC_DIR)/npu/spec_sampler.h \
            $(SRC_DIR)/npu/npu_window.h $(SRC_DIR)/npu/common.h $(SRC_DIR)/chat_callback.h
$(BUILD_DIR)/chat: examples/chat.cpp $(SPEC_SRCS) $(LIB_SRCS) $(ENGINE_DEPS) $(SPEC_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXX_FAST) $(INCLUDES) -o $@ $< $(SPEC_SRCS) $(LIB_SRCS) $(LDFLAGS) $(OV_LDFLAGS) -lpthread

.PHONY: chat
chat: $(BUILD_DIR)/chat

# ── NPU spec example (requires OpenVINO + NPU blob) ──────────────
# $(BUILD_DIR)/spec_npu: src/npu/spec.cpp $(LIB_SRCS) $(ENGINE_DEPS) | $(BUILD_DIR)
#	$(CXX) $(CXX_FAST) $(INCLUDES) -o $@ $< $(LIB_SRCS) $(LDFLAGS) -lopenvino
#
# .PHONY: spec_npu
# spec_npu: $(BUILD_DIR)/spec_npu

# ── Clean ─────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(LIB_NAME) $(LIB_NAME_BF16)
