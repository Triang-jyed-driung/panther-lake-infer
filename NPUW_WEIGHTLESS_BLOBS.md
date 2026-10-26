# OpenVINO NPUW 无权重 blob 与共享权重方案

日期：2026 年 10 月 26 日。

在本机 Panther Lake NPU 上，RWKV 0.1B 已完成 NPUW 无权重 blob 的导出、重新加载和推理验证。同一份 IR 权重 `.bin` 也已供 1、4、8-token 三个独立 blob 使用，并通过 C++ 同时加载及交替推理测试。

当前可用方案采用显式循环状态输入输出，尚未接入应用的 `NpuWindow`。应用 C++ 源码未修改；本次新增的 C++ 程序仅用于独立实验。

## 测试环境

| 项目 | 本机环境 |
| --- | --- |
| CPU | Intel Core Ultra 5 325 |
| NPU | Panther Lake NPU5010，PCI ID `8086:b03e` |
| 系统 | Linux x86_64，内核 `7.1.5-070105-generic` |
| NPU 驱动包 | `1.33.0.20260529-26625960453~ubuntu24.04` |
| OpenVINO | `2026.3.1-22476-759c5a6ab8c-releases/2026/3` |
| OpenVINO GenAI | `2026.3.1.0-3290-56d9685302d` |
| 测试 checkpoint | `/D/g1d_0.1b.pth` |
| 模型维度 | 12 层，宽度 768，12 个头，每头 64，词表 65536 |

本文中的 `.bin` 需要区分三种用途：OpenVINO IR 的常量权重文件、CPU 引擎的模型权重文件，以及宿主查表使用的融合嵌入文件。`WEIGHTS_PATH` 必须指向第一种，不能填 CPU 引擎的 `.bin`。

## 无权重 blob 的机制

无权重 blob 将已编译的计算与外部模型权重分开保存。blob 仍包含计算代码、调度、图连接、权重来源和变换信息，也可能保留部分生成的常量。

NPUW 在函数化、折叠等处理过程中，将适合外置的权重从 Constant 提取为函数输入。运行时由 NPUW 管理权重张量，并绑定到相应的输入位置：

```text
原图                         编译后的函数
y = MatMul(x, Constant W)  →  y = MatMul(x, W_input)
                                  ↑
                       从原始 IR .bin 取得并绑定权重
```

对于能追溯到原始 IR 的权重，无权重序列化保存数据类型、形状、字节数和原始 `.bin` 偏移。`LazyTensor` 还可保存转换、排列、解包等操作，以便导入时重建张量。不能追溯到外部权重的常量可继续随 blob 保存。[序列化实现](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/plugin/npuw/serialization.cpp#L303)、[LazyTensor 实现](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/plugin/npuw/lazy_tensor.cpp#L440)。

导入可通过 `WEIGHTS_PATH` 指定原始 IR 权重文件，或通过 `MODEL_PTR` 提供含有权重来源信息的模型。文件可用 mmap 访问，但后续仍可能发生复制、解包、转换及设备内存分配；无权重文件并不意味着运行时没有权重内存开销。[导入实现](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/plugin/npuw/compiled_model.cpp#L114)。

本次 NPUW 导出文件以 `NPUWORC` 开头，是包含编译子图和调度信息的封装；普通 NPU 路径的测试文件以 ELF 开头。NPUW blob 不能整体交给裸 Level Zero 图加载器，须先经过 OpenVINO NPUW 的反序列化和权重绑定。

## 小模型对照实验

使用一个 FP16 的 `512 × 512` 矩阵作为 MatMul 权重，原始权重大小为 524,288 字节。

| 编译路径 | 缓存模式 | blob 字节数 | 不提供外部权重 | 提供原始权重 |
| --- | --- | ---: | --- | --- |
| 普通 NPU | `OPTIMIZE_SPEED` | 573,522 | 成功 | 成功 |
| 普通 NPU | `OPTIMIZE_SIZE` | 573,522 | 成功 | 成功 |
| NPUW | `OPTIMIZE_SPEED` | 590,689 | 成功 | 成功 |
| NPUW | `OPTIMIZE_SIZE` | 66,434 | 导入失败 | 推理结果逐位一致 |

同一个 NPUW 无权重 blob 分别导入原权重和两倍数值的权重，独立进程运行后的输出也精确变为两倍。这验证了该 MatMul 的外部权重绑定，不构成任意模型权重可替换的保证。

普通 NPU 插件也有独立的权重分离实现：编译器可生成权重初始化图和主计算图，初始化图生成的权重缓冲区再绑定到主图。对本次简单 MatMul，指定 `ENABLE_WEIGHTLESS=YES`，或指定 `CACHE_DIR` 加 `OPTIMIZE_SIZE`，均未实际分离权重；日志显示 `Total number of Inits: 0`。因此不能只凭配置值认定导出文件已无权重。[底层实现](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/compiler_adapter/src/weightless_graph.cpp#L425)。

原始记录：[对照结果](benchmark_results/npu_weightless/results.json)、[普通路径补充测试](benchmark_results/npu_weightless/native_results.json)、[替换权重测试](benchmark_results/npu_weightless/rebind_results.json)。

## RWKV 0.1B 的无权重导出

### 编译配置

本次直接通过 OpenVINO Core 使用 NPUW，没有将 RWKV 接入 GenAI 的 LLM pipeline。

```python
config = {
    "NPU_USE_NPUW": "YES",
    "NPUW_DEVICES": "NPU",
    "NPUW_FALLBACK_EXEC": "NO",
    "NPUW_ONLINE_PIPELINE": "NONE",
    "NPUW_FOLD": "YES",
    "NPUW_FUNCALL_FOR_ALL": "YES",
    "CACHE_MODE": "OPTIMIZE_SIZE",
}
```

限定 NPU 设备并禁用执行回退，用于确认测试没有依靠 CPU 回退完成。`NONE` 分区配置用于先验证外置权重，不代表已完成重复层折叠或性能调优。这些 NPUW 属性属于内部接口，需固定运行库和设备版本。

### 循环状态接口

原有 4-token 图使用 `ReadValue/Assign` 管理每层的 `xp`、`wkv` 和 `fp`，共 36 个循环状态。该图可以导出 NPUW blob，但创建推理请求时报错：

```text
The link between state descriptors is missing, state name: L0.wkv
```

改用 `REG` 分区流程时，编译以 `map::at` 错误失败。本机已验证可用的路径是把循环状态改成显式输入输出：

| 接口 | 内容 |
| --- | --- |
| 输入 0 | 融合嵌入 `[n, 768]`，FP32 |
| 输入 1 至 36 | 每层的三个循环状态，FP32 |
| 输出 0 至 3 | logits、att 快照、wkv 快照、ffn 快照 |
| 输出 4 至 39 | 下一次调用需要的 36 个循环状态 |

第一轮输入零状态；推理后，将状态输出回传为下一轮输入。输入输出对应关系由 `.states.json` 记录。显式状态版本不再通过 `query_state()` 访问循环状态。

转换工具只处理本项目具有隐式零初始化的 `ReadValue/Assign` 配对，不能视为任意有状态模型的通用转换器。

### 导出与加载结果

| 项目 | 字节数 | MiB |
| --- | ---: | ---: |
| IR 权重 | 281,754,820 | 268.70 |
| 4-token NPUW 无权重 blob | 23,561,737 | 22.47 |

blob 大小比 IR 权重文件小约 91.64%。这是文件大小对比，完整部署仍需外部权重和约 96 MiB 的宿主融合嵌入表。

- 原有状态图与显式状态图在 CPU 上连续执行两个窗口，四类原始输出和全部循环状态逐位一致。
- NPUW 编译对象与独立进程导入的对象连续执行两个窗口，80 个输出张量逐位一致，输出均为有限值。
- 不提供 `WEIGHTS_PATH` 或 `MODEL_PTR` 时，导入按预期失败。
- 相比普通 NPU 编译路径，NPUW 输出存在数值差异：8 个测试位置的 top-1 相同，logits 最大绝对差为 0.203125。导出与导入的一致性不等同于不同编译路径的一致性。

短测中的第二个 4-token 窗口，NPUW 用时约 27.36 ms，普通 NPU 路径约 12.73 ms。它们只是单次观测，不是稳定吞吐基准；本次配置没有证明推理加速。

产物和记录：[无权重 blob](scripts/npu/export/npuw_g1d_0.1b_n4_external.blob)、[IR 权重](scripts/npu/export/window_v6_g1d_0.1b_n4_external.bin)、[状态映射](scripts/npu/export/window_v6_g1d_0.1b_n4_external.states.json)、[汇总报告](benchmark_results/npu_weightless/rwkv_0.1b_summary.json)、[状态转换校验](benchmark_results/npu_weightless/rwkv_state_conversion_check.json)。

### 复现命令

以下命令在本仓库根目录运行，checkpoint 路径使用当前机器的 `/D/g1d_0.1b.pth`。

```bash
python3 scripts/npu/build_window_graph.py /D/g1d_0.1b.pth 4

python3 scripts/npu/externalize_rwkv_state.py \
  scripts/npu/export/window_v6_g1d_0.1b_n4.xml \
  scripts/npu/export/window_v6_g1d_0.1b_n4_external.xml

python3 scripts/npu/npuw_weightless.py export \
  --model scripts/npu/export/window_v6_g1d_0.1b_n4_external.xml \
  --blob scripts/npu/export/npuw_g1d_0.1b_n4_external.blob \
  --emb scripts/npu/export/emb_g1d_0.1b.f16.bin

python3 scripts/npu/npuw_weightless.py verify \
  --model scripts/npu/export/window_v6_g1d_0.1b_n4_external.xml \
  --blob scripts/npu/export/npuw_g1d_0.1b_n4_external.blob \
  --emb scripts/npu/export/emb_g1d_0.1b.f16.bin
```

`export` 与 `verify` 分别启动进程。将最后一条命令中的 `verify` 换成 `check-missing-weights`，可检查缺失外部权重时的拒绝行为。`WEIGHTS_PATH` 由此工具根据 `--model` 推导为同名 IR `.bin`。

## 一份 bin 对应多个 blob

### 共用的条件

多个 blob 可以引用同一份权重文件，但它们保存的偏移、大小、类型及变换信息必须与这份文件相符。来自同一个 checkpoint，并不保证分别序列化的 `.bin` 布局相同。

本次从 `.pth` 分别生成 1、4、8-token IR 后，三个原始 `.bin` 的布局确实不同。若仅把它们的 `WEIGHTS_PATH` 改为同一路径，不能保证取得正确的权重。

### 已验证的统一布局方案

实验采用以下过程，在编译之前统一常量来源：

1. 以 4-token 显式状态 IR 的 `.bin` 为基础建立 `shared_weights.bin`。
2. 按常量字节数和内容哈希查找已有数据，将其他窗口 IR 的 Constant 偏移改为公共文件中的位置。
3. 公共文件中尚不存在的常量追加到末尾，1-token 补入 56 字节，8-token 补入 88 字节，总共增加 144 字节。
4. 使用 `core.read_model(variant_xml, shared_weights_bin)` 读取每个图，再分别通过 NPUW 编译和导出。
5. 导入三个 blob 时，全部传入同一个 `WEIGHTS_PATH`。

重映射后实际逐字节检查了 6,594 个常量引用，数据均与原图一致。这种方案保留计算图的类型和形状定义，只调整常量在权重文件中的位置。

### 多窗口测试结果

公共权重文件大小为 281,754,964 字节。三个 blob 分别为：

| 窗口大小 | blob 字节数 | MiB | 独立进程导入校验 |
| --- | ---: | ---: | --- |
| 1 token | 18,933,221 | 18.06 | 逐位一致 |
| 4 tokens | 23,561,737 | 22.47 | 逐位一致 |
| 8 tokens | 27,694,608 | 26.41 | 逐位一致 |

独立 C++ 程序还完成了以下测试：三个编译对象和推理请求同时驻留，每个对象保留独立循环状态，交替执行两个窗口。总共 240 个输出张量与各自编译时保存的参考结果逐位一致。不提供外部权重时，导入按预期失败。

这验证了文件层面的共用，以及三个对象在本次加载和销毁顺序下共存。跨窗口切换同一段生成过程的状态、不同窗口之间的浮点结果一致性，以及设备权重缓冲区是否共用，未包含在该测试中。

可供导入的文件集中在：

```text
benchmark_results/npu_weightless/shared_bin/deploy/
    shared_weights.bin
    window_n1.blob
    window_n4.blob
    window_n8.blob
```

该目录在本机使用硬链接整理产物。移动到其他机器时可复制实际文件。宿主融合嵌入表依然单独提供；它没有包含在上述公共 IR 权重文件中。正式接入还需要状态映射或等价的接口元数据。

详细记录：[汇总报告](benchmark_results/npu_weightless/shared_bin/summary.json)、[布局准备记录](benchmark_results/npu_weightless/shared_bin/prepare.json)、[C++ 测试日志](benchmark_results/npu_weightless/shared_bin/cpp_import.log)。

### 复现共享权重实验

先完成前面的 4-token IR 准备，再运行：

```bash
npuw_probe_dir=benchmark_results/npu_weightless/shared_bin

python3 "$npuw_probe_dir/probe.py" prepare

for npuw_window in 1 4 8; do
  python3 "$npuw_probe_dir/probe.py" compile "$npuw_window"
  python3 "$npuw_probe_dir/probe.py" verify "$npuw_window"
done
```

`probe.py` 是本机实验脚本，包含当前仓库和 checkpoint 的绝对路径。`prepare` 会重建实验目录中的 IR 和公共权重；改变布局后，应重新编译相应的 blob。

## C++ 导入方式

OpenVINO C++ 直接支持通过 `WEIGHTS_PATH` 导入无权重 NPUW blob。多个编译对象可传入相同的权重路径：

```cpp
#include <openvino/openvino.hpp>
#include <fstream>
#include <stdexcept>
#include <string>

ov::Core core;
const std::string weights_path = "shared_weights.bin";
const ov::AnyMap properties{{"WEIGHTS_PATH", weights_path}};

std::ifstream stream1("window_n1.blob", std::ios::binary);
std::ifstream stream4("window_n4.blob", std::ios::binary);
if (!stream1 || !stream4) {
    throw std::runtime_error("Cannot open NPUW blobs");
}

auto model1 = core.import_model(stream1, "NPU", properties);
auto model4 = core.import_model(stream4, "NPU", properties);
auto request1 = model1.create_infer_request();
auto request4 = model4.create_infer_request();
```

上述片段完成导入。推理前还需填入嵌入和初始状态，推理后回传循环状态。本次转换工具生成的接口中，状态输入索引 `i` 对应状态输出索引 `i + 3`，其中 `i` 为 1 至 36；一般实现应读取状态映射，而不能将此关系用于任意模型。

完整独立测试程序：[import_multiple.cpp](benchmark_results/npu_weightless/shared_bin/import_multiple.cpp)。在当前机器上可按以下方式编译和运行：

```bash
npuw_probe_dir=benchmark_results/npu_weightless/shared_bin
npuw_ov_root="$(python3 -c 'import pathlib, openvino; print(pathlib.Path(openvino.__file__).parent)')"

g++ -std=c++17 -O2 \
  -I"$npuw_ov_root/include" \
  "$npuw_probe_dir/import_multiple.cpp" \
  -L"$npuw_ov_root/libs" -l:libopenvino.so.2631 \
  -Wl,-rpath,"$npuw_ov_root/libs" \
  -o "$npuw_probe_dir/import_multiple"

"$npuw_probe_dir/import_multiple" \
  "$npuw_probe_dir/artifacts" \
  scripts/npu/export/emb_g1d_0.1b.f16.bin
```

测试程序从 `artifacts` 读取三个 blob、公共权重及参考输出文件，因此不能直接把测试命令中的目录换成只有部署文件的 `deploy`。

现有 [npu_window.h](src/npu/npu_window.h) 导入时未传入外部权重，并依赖 `query_state()` 管理状态。接入上述 blob 需要同时适配导入属性、状态反馈、复位和回退逻辑；只有增加 `WEIGHTS_PATH` 不足以完成接入。应用源码尚未进行这些修改。

## GenAI 如何保留同一份权重来源

GenAI 的 NPU 路径设置 `NPU_USE_NPUW` 和 `NPUW_LLM`，再调用 OpenVINO 的编译、导出及导入接口。[GenAI 配置入口](https://github.com/openvinotoolkit/openvino.genai/blob/56d9685302da2fc5cc7c9689cfab500fd0660a02/src/cpp/src/utils.cpp#L79)。

NPUW 的 `LLMCompiledModel` 从同一个传入模型克隆 decode 图，再克隆 prefill 图，并可派生不同 KV 长度的版本。其外部权重引用依赖原始 IR 的 `WeightlessCacheAttribute` 和文件偏移，而非为每个版本另行分配一套权重文件位置。[图派生代码](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp#L813)、[偏移保存代码](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/plugin/npuw/compiled_model.cpp#L1479)。

对外一个 GenAI blob 也可以封装多个内部编译对象，包括多个 decode 版本、prefill 和可选的 LM head。因而“一份 bin 对应多个计算图”不一定表现为磁盘上多个独立 `.blob` 文件。[封装实现](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp#L1330)。

GenAI 的这种派生方式通常不需要本次实验的事后偏移重映射。后者是为了解决 RWKV 从 `.pth` 分别重建、序列化不同窗口 IR 后产生的布局差异。

### 文件共用与运行时权重共用

同一个 LLM 对象为 prefill、decode 配置共同的 Weights Bank；导入时也将相关子模型关联到共同的 bank。bank 可以复用匹配的权重张量及相应变换结果。[bank 配置](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp#L1040)、[导入时关联 bank](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c066af5f4bc5ebd04643706012a37d/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp#L1432)。

文件路径和 bank 身份是不同概念。多个独立 pipeline 或编译对象使用相同 `WEIGHTS_PATH`，并不自动保证所有设备权重分配都共用。本次 RWKV 多 blob 测试验证了共用权重文件，没有测定设备内存去重效果。

后续若将方案正式用于 RWKV，较合适的构建方式是在生成不同窗口图时保留统一的权重来源和偏移元数据，避免反复重新分配常量布局。状态切换和设备内存共享需要分别验证。

## npunlock 的兼容性发现

npunlock 的自定义 SHAVE 内核路径与无权重缓存是不同能力。本机 Linux + Panther Lake 上，项目的 Linux 工具编译成功，27 项离线测试通过，但仓库自带的 Meteor Lake NPU3720 加一内核 blob 在图创建时被拒绝：

```text
0x7800000f  ZE_RESULT_ERROR_INVALID_NATIVE_BINARY
```

作为对照，本机编译的普通 Abs 图能通过相同的 Linux 执行探测工具在 NPU 上运行，16 个结果全部正确。因此已确定旧 blob 不适用于当前设备和软件栈；失败发生在图创建阶段，不能据此认定 SHAVE 指令集本身不兼容。

NPUW 无权重导出依靠当前设备支持的常规编译和执行路径，不要求先完成自定义 SHAVE 内核移植。[npunlock 实测记录](benchmark_results/npunlock_compat/summary.json)。

## 使用边界

- 无权重 blob 仍依赖正确版本的外部权重、兼容的设备、驱动和运行库，不是任意同结构模型的通用程序。
- 公共 `.bin` 与 blob 必须作为一致的一组产物管理；不能在修改权重布局后继续使用保存旧偏移的 blob。
- 本次 RWKV 方案使用显式状态，并保留单独的宿主嵌入表；现有 C++ 应用尚不能直接换用这些 blob。
- 早期小矩阵实验在同进程替换活跃编译对象时发生过段错误。后续 RWKV C++ 三对象同时驻留、先销毁请求再销毁模型的测试通过，验证范围不包括任意卸载顺序或跨线程并发。
- 完整生成质量、跨窗口状态接续、持续吞吐、7B 模型兼容性和设备内存去重均不能由本次短测结果推定。
