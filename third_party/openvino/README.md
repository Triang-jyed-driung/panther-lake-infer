# OpenVINO 公共头文件（vendor）

来源：https://github.com/openvinotoolkit/openvino tag **2026.3.1**，
目录 `src/core/include/openvino` 与 `src/inference/include/openvino` 的合并
（纯公共头文件，无任何源码/第三方依赖）。

版本必须与链接的 `libopenvino.so.2631`（pip wheel openvino==2026.3.1）一致，
也与 NPU blob（export/*.blob，OpenVINO 2026.3.1 编译）一致，不可单独升级。

- 头文件 include 路径：`third_party/openvino/include`
- 运行库：pip wheel 的 `site-packages/openvino/libs`（含 NPU 插件），Makefile 自动探测
