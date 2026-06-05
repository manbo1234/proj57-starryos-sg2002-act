# proj57 StarryOS/QEMU ACT 推理部署

本项目完成了 `chenlongos/proj57` 赛题任务三：在 `StarryOS + QEMU + CPU` 上运行 ACT 模型推理。

## 基本信息

| 项目 | 内容 |
|------|------|
| 平台 | StarryOS / QEMU |
| 架构 | riscv64 |
| 推理后端 | ONNX Runtime CPU |
| 推理程序语言 | C |
| 交叉工具链 | riscv64-linux-musl |
| 模型格式 | ONNX |

核心程序为 `act_infer.c`，通过 ONNX Runtime C API 加载 ONNX 模型，读取预处理后的 `image/state/latent` 输入，执行推理并输出左右轮速度和方向。

## 完成进度

| 任务 | 平台 | 推理后端 | 状态 |
|------|------|----------|------|
| 任务三 | StarryOS / QEMU | CPU / ONNX Runtime | √ 已完成 |
| 任务二 | RK3588 | NPU / RKNN Toolkit2 | × 未完成 |
| 任务一 | SG2002 | TPU / Sophon TPU SDK | × 未完成 |

当前提交重点覆盖任务三，即在 QEMU 中运行的 StarryOS 上完成 ACT 模型 CPU 推理，并输出与参考一致的左右转方向。

## 主要文件

```text
src/act_infer.c                  # StarryOS 侧 C 推理程序
scripts/freeze_reference.py      # 固定 PyTorch 参考输出
scripts/export_verify_onnx.py    # 导出并校验 ONNX
scripts/prepare_starry_inputs.py # 生成 StarryOS 使用的 .bin 输入
```

运行所需文件：

```text
act_infer
proj57_deterministic.onnx
proj57_deterministic.onnx.data
frame_000000_input.bin
frame_000227_input.bin
state.bin
latent.bin
libonnxruntime.so*
必要 musl 运行库
```

## 部署目录

Linux VM 侧：

```text
~/proj57/
├── proj57_deterministic.onnx
├── proj57_deterministic.onnx.data
└── starry_inputs/
    ├── frame_000000_input.bin
    ├── frame_000227_input.bin
    ├── state.bin
    └── latent.bin

~/StarryOS/
├── act_infer
├── act_infer.c
└── libonnxruntime.so
```

StarryOS 侧：

```text
/root/proj57/
├── act_infer
├── proj57_deterministic.onnx
├── proj57_deterministic.onnx.data
├── frame_000000_input.bin
├── frame_000227_input.bin
├── state.bin
├── latent.bin
└── libonnxruntime.so*
```

系统运行库需放到 StarryOS 可加载的位置，例如 `/lib/`：

```text
ld-musl-riscv64.so.1
libatomic.so.1
libgcc_s.so.1
libstdc++.so.6
```

## 运行方式

在 StarryOS 中执行：

```bash
cd /root/proj57
./act_infer proj57_deterministic.onnx frame_000000_input.bin state.bin latent.bin
./act_infer proj57_deterministic.onnx frame_000227_input.bin state.bin latent.bin
```

## 验证结果

| 输入 | left_vel | right_vel | direction |
|------|----------|-----------|-----------|
| `frame_000000_input.bin` | `-0.001586` | `+0.007239` | `left` |
| `frame_000227_input.bin` | `+0.006118` | `-0.000454` | `right` |

结果与 Linux/PyTorch 参考输出方向一致，说明模型加载、外部权重读取、输入 tensor 构造、ONNX Runtime 推理和动作后处理均已跑通。

## 文件大小

| 文件 | 大小 |
|------|------|
| `act_infer` | `13K` |
| `act_infer.c` | `5.2K` |
| `libonnxruntime.so` | `23M` |
| `rootfs-riscv64.img` | `1.0G` |

## 说明

当前实现采用离线预处理方案：图像 resize、normalize、state 归一化和 latent 固定在 Linux VM 中完成，StarryOS 侧直接读取 float32 `.bin` 输入并执行推理。


## AI 使用说明

本项目开发过程中使用了 AI 辅助，主要用于：

- 梳理从 PyTorch 到 ONNX，再到 StarryOS/QEMU 的迁移路线
- 辅助编写和修改 `freeze_reference.py`、`export_verify_onnx.py`、`prepare_starry_inputs.py`
- 辅助分析 ONNX Runtime 交叉编译错误、动态库依赖和运行时报错
- 辅助编写 `act_infer.c` 中 ONNX Runtime C API 的接入流程
- 辅助整理部署说明、运行结果和提交文档

人工完成和确认的部分包括：

- 在 Linux VM 中实际下载、安装、编译和运行相关工具
- 在 StarryOS/QEMU 中实际部署和执行 `act_infer`
- 根据终端输出确认左右转推理结果
- 检查最终需要提交的源码、模型、输入文件和运行库

本项目没有直接复制其他参赛者的完整实现。原始模型结构、数据格式、checkpoint 和 Notebook 参考流程来自赛题仓库 `chenlongos/proj57`，StarryOS 运行环境来自 StarryOS 项目，ONNX Runtime 来自 Microsoft ONNX Runtime 开源项目。
