# proj57 StarryOS SG2002 ACT 推理部署

本仓库用于记录和复现 `chenlongos/proj57` 赛题中 ACT 模型在 SG2002 / LicheeRV Nano 开发板上的部署过程。项目从 StarryOS/QEMU 上的 CPU 推理验证开始，逐步迁移到 SG2002 官方 Linux 与 StarryOS 上的 TPU 推理，最终在 SG2002 开发板的 StarryOS 中跑通 ACT BF16 `cvimodel` 推理。

## 项目状态

| 阶段 | 状态 |
|---|---|
| StarryOS/QEMU CPU 推理 | 已完成 |
| SG2002 开发板启动 StarryOS | 已完成 |
| Docker / TPU-MLIR BF16 模型转换 | 已完成 |
| SG2002 官方 Linux TPU 验证 | 已完成 |
| StarryOS `/dev/cvi-tpu0` 与 `/dev/ion` 支持 | 已完成 |
| StarryOS 官方 TPU 小模型运行 | 已完成 |
| StarryOS ACT BF16 TPU 推理 | 已完成 |

## 技术路线

```text
StarryOS/QEMU CPU 推理
-> SG2002 开发板启动 StarryOS
-> TPU-MLIR 将 ACT ONNX 转为 BF16 cvimodel
-> SG2002 官方 Linux 验证 TPU 推理正确性
-> 在 StarryOS 中补齐 TPU/ION 设备支持
-> 在 SG2002 StarryOS 中运行 ACT BF16 TPU 推理
```

## 仓库内容

```text
src/                         # StarryOS/QEMU CPU 推理程序
scripts/                     # ONNX 导出、输入准备、TPU-MLIR 前处理脚本
onnxruntime_patches/          # ONNX Runtime riscv64 交叉编译相关补丁
patches/tgoskits/             # StarryOS / tgoskits 的 SG2002 TPU/ION 修改补丁
docs/                         # Docker 复现和运行说明
models/                       # 大模型文件说明，不直接提交模型
data/                         # 输入数据说明，不直接提交生成数据
artifacts/                    # 生成产物说明，不直接提交产物
```

## Docker 复现方式

本仓库使用 `sophgo/tpuc_dev:latest` 作为基础镜像。该镜像本身不包含完整 TPU-MLIR 工作树，因此复现时需要挂载本机已经准备好的 `tpu-mlir` 目录。

已验证环境：

```text
base image : sophgo/tpuc_dev:latest
image id   : d46a29a349f1
project    : /workspace/proj57
tpu-mlir   : /workspace/tpu-mlir
```

构建镜像：

```bash
docker build -t proj57-sg2002-act:latest .
```

进入容器：

```bash
docker run --rm -it \
  -v "$PWD:/workspace/proj57" \
  -v "$HOME/tpu-mlir:/workspace/tpu-mlir" \
  proj57-sg2002-act:latest
```

初始化 TPU-MLIR：

```bash
cd /workspace/tpu-mlir
source envsetup.sh
```

检查工具链：

```bash
which model_transform.py
which model_deploy.py
which npz_tool.py
```

已验证输出：

```text
/workspace/tpu-mlir/python/tools/model_transform.py
/workspace/tpu-mlir/python/tools/model_deploy.py
/workspace/tpu-mlir/python/tools/npz_tool.py
```

## 模型转换

模型文件不直接提交到 Git。复现时将模型放入：

```text
models/proj57_relu_tpu_fold2.onnx
models/proj57_relu_tpu_fold2.onnx.data
```

转换命令：

```bash
cd /workspace/proj57

model_transform.py \
  --model_name act_relu \
  --model_def /workspace/proj57/models/proj57_relu_tpu_fold2.onnx \
  --input_shapes [[1,1,3,224,224],[1,2],[1,32]] \
  --output_names action \
  --mlir act_relu.mlir

model_deploy.py \
  --mlir act_relu.mlir \
  --quantize BF16 \
  --chip cv181x \
  --model act_relu_bf16.cvimodel
```

已验证的输入输出形状：

```text
image  : [1, 1, 3, 224, 224]
state  : [1, 2]
latent : [1, 32]
action : [1, 8, 3]
```

生成的 `act_relu_bf16.cvimodel`、`.mlir`、`.npz` 等文件不提交到 Git，可放入 GitHub Releases 或复制到 SG2002 板端验证。

## StarryOS 修改

本仓库不提交完整 `tgoskits` 源码树，只提交补丁：

```text
patches/tgoskits/sg2002-starryos-tpu-ion-carveout.patch
```

应用方式：

```bash
git clone --recursive -b dev https://github.com/rcore-os/tgoskits.git
cd tgoskits
git apply ../proj57-starryos-ACT/patches/tgoskits/sg2002-starryos-tpu-ion-carveout.patch
cargo starry quick-start licheerv-nano-sg2002 build
```

补丁主要包含：

```text
/dev/cvi-tpu0
/dev/ion
IonBufferFile mmap
/proc/self/fdinfo/<fd>
SG2002 ION carveout 内存保留
ION_IOC_FREE 释放路径
```

## 板端运行

在 SG2002 StarryOS 中运行：

```bash
cd /root/proj57_tpu_test

/usr/bin/model_runner \
  --model ./act_relu_bf16.cvimodel \
  --input ./act_test_000000.npz \
  --output ./act_starry_000000.npz \
  --count 1 \
  --enable-timer \
  --verbose 1
```

关键输出示例：

```text
Inputs:
  [0] image <1,1,3,50176>,fp32
  [1] state <1,2,1,1>,fp32
  [2] latent <1,32,1,1>,fp32
Outputs:
  [0] action_Add_f32 <1,8,3,1>,fp32
Task exit with code: 0
```

性能记录（优化前，TPU 热路径 info 日志开启）：

```text
run1: 239.647 ms, 4.172804 FPS
run2: 239.683 ms, 4.172177 FPS
avg : 239.665 ms, 4.1725 FPS
```

### StarryOS TPU 热路径日志优化

进一步用 `--count 3 --enable-timer --verbose 0` 测试时发现，`--verbose 0` 只能关闭 `model_runner` 用户态输出，不能关闭 StarryOS 内核 TPU 设备路径中的 `info!` 日志。优化前，每次推理都会在热路径中输出：

```text
TPU ioctl enter
TPU cache_flush
TPU cache_flush done
TPU cache_invalidate
TPU cache_invalidate done
```

这些串口日志位于每轮 TPU 推理的 cache flush、任务提交和 cache invalidate 路径中，会被计入 `model_runner` 的单次推理耗时。优化方式是在 `tgoskits` 的：

```text
os/StarryOS/kernel/src/pseudofs/dev/tpu/device.rs
```

中将上述 TPU 热路径日志从 `info!` 降为 `debug!`，只修改日志级别，不修改 TPU 提交、cache flush 或 cache invalidate 的实际逻辑。

优化前后性能对比：

```text
before: 3 runs take 713.305 ms, each run takes 237.768000 ms, fps 4.205780
after : 3 runs take 253.313 ms, each run takes  84.437000 ms, fps 11.843149
```

因此，StarryOS 上此前约 240 ms 的 ACT BF16 推理耗时主要由 TPU 热路径内核日志污染导致。关闭热路径 `info!` 日志后，ACT BF16 `cvimodel` 在 SG2002 StarryOS 上的推理速度达到约 84 ms/run，接近 SG2002 官方 Linux 上约 83 ms/run 的结果。

## 板端内存占用

SG2002 StarryOS 当前可见内存约为：

```text
MemTotal: 241424 kB
MemFree : 231867 kB
```

在 QEMU / VMware 中使用完整 ONNX Runtime CPU 路线运行 ACT 时，峰值内存约为：

```text
253-256 MB
```

因此，直接把完整 ONNX Runtime CPU 推理搬到 SG2002 StarryOS 上存在明显 OOM 风险。本项目最终改为 TPU-MLIR BF16 路线，将 ACT 模型转换为 SG2002 TPU 可运行的 `cvimodel`。

在 StarryOS SG2002 上运行 ACT BF16 `cvimodel` 时，模型需要一次较大的 ION carveout buffer 申请：

```text
ION alloc_buffer request: size=98799760, align=1, heap_type=Carveout
```

该申请大小约为：

```text
98,799,760 bytes ~= 94.2 MiB
```

这也是移植过程中定位到的关键内存问题：普通 DMA heap fallback 无法稳定承载这类大块连续内存申请，需要在 StarryOS 侧为 SG2002 保留并接入 ION carveout 内存区域。当前补丁中保留的 SG2002 ION carveout 物理内存范围为：

```text
base: 0x8950_0000
size: 0x0690_0000
end : 0x8fe0_0000
```

## 大文件说明

以下文件不提交到 Git：

```text
*.onnx
*.onnx.data
*.cvimodel
*.mlir
*.npz
*.bin
*.so
boot.sd
rootfs*
```

这些文件应由脚本生成，或通过 GitHub Releases 发布。

## 已知限制

当前已跑通 ACT BF16 `cvimodel` 在 SG2002 StarryOS 中的 TPU 推理，但仍有后续工作可以继续完善：

1. 将 StarryOS 生成的 `act_starry_000000.npz` 回拷到 Docker / VM 中，与 BF16 仿真输出做 `npz_tool.py compare`。
2. 对 `act_test_000227.npz` 做同样的 StarryOS 实机推理和输出比较。
3. 将 `action [1,8,3]` 接回 ACT 后处理逻辑，恢复最终 left/right 方向判断。
4. 继续整理 StarryOS 侧 TPU / ION / carveout 修改，形成更干净的补丁。
5. 后续实现更完整的 carveout heap。
6. 已完成 TPU 热路径日志降级优化，将 StarryOS 上 ACT BF16 推理从约 240 ms/run 降至约 84 ms/run；后续仍可继续减少模型加载、ION buffer 分配、文件 IO 和非热路径日志开销。

## AI 工具使用说明

本项目开发过程中使用了 AI 工具作为辅助，主要用于技术路线梳理、报错信息分析、命令记录整理、README 文档草拟以及部分脚本修改建议。项目中的环境搭建、模型转换、StarryOS 构建、TF 卡部署、SG2002 上板运行、串口输出采集、推理结果对比和最终结论均由本人实际执行、检查和确认。

AI 工具生成或建议的内容均经过人工筛选、修改和验证，未直接作为未经验证的最终结果提交。
