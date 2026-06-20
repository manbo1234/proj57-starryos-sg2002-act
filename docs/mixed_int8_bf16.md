# ACT mixed INT8/BF16 在 SG2002 StarryOS 上的部署验证

## 环境

- 开发板：SG2002 / LicheeRV Nano
- 操作系统：StarryOS
- TPU-MLIR 容器：`sophgo/tpuc_dev:latest`
- 虚拟机项目路径：`/home/whiecloud/proj57`
- 容器内项目路径：`/workspace/proj57`
- 芯片参数：`cv181x`

## 数据准备

ACT 模型有三个输入：

```text
image  : [1, 1, 3, 224, 224]
state  : [1, 2]
latent : [1, 32]
```

使用 `export_act_npz.py` 导出 TPU-MLIR 需要的测试样本和校准样本：

```text
test npz files  : 2
calibration npz : 128
```

生成的数据文件不提交到 Git：

```text
artifacts/tpu_mlir_npz/act_test_000000.npz
artifacts/tpu_mlir_npz/act_test_000227.npz
artifacts/tpu_mlir_npz/act_cali_dataset/
```

## 转换为 MLIR

在 TPU-MLIR 容器中执行：

```bash
model_transform.py \
  --model_name act_relu \
  --model_def /workspace/proj57/proj57_relu_tpu_fold2.onnx \
  --input_shapes '[[1,1,3,224,224],[1,2],[1,32]]' \
  --output_names action \
  --mlir act_relu.mlir
```

转换完成后，对 ONNX 输出和 MLIR 输出做数值对齐检查：

```text
000000 max_abs_diff ~= 9.54e-07
000227 max_abs_diff ~= 1.07e-06
```

这说明 ONNX 到 MLIR 的转换没有引入明显误差。

## 混合量化

先生成校准表：

```bash
run_calibration.py act_relu.mlir \
  --dataset /workspace/proj57/artifacts/tpu_mlir_npz/act_cali_dataset \
  --input_num 128 \
  --chip cv181x \
  --cali_method kl \
  -o act_relu_cali_table
```

再搜索 mixed INT8/BF16 量化表：

```bash
run_calibration.py act_relu.mlir \
  --dataset /workspace/proj57/artifacts/tpu_mlir_npz/act_cali_dataset \
  --input_num 128 \
  --chip cv181x \
  --cali_method kl \
  --search search_qtable \
  --fp_type BF16 \
  --expected_cos 0.995 \
  --min_layer_cos 0.99 \
  --max_float_layers 12 \
  -o act_relu_cali_table \
  --quantize_table act_relu_mix_qtable
```

搜索结果：

```text
int8 outputs_cos: 0.984688
float layer number: 12
mix model outputs_cos: 0.9999107254577259
success search qtable
```

从结果可以看出，纯 INT8 的输出相似度不足，而将部分层提升为 BF16 后，输出相似度恢复到 `0.9999` 以上。

部署生成 mixed INT8/BF16 `cvimodel`：

```bash
model_deploy.py \
  --mlir act_relu.mlir \
  --quantize INT8 \
  --calibration_table act_relu_cali_table \
  --quantize_table act_relu_mix_qtable \
  --chip cv181x \
  --test_input /workspace/proj57/artifacts/tpu_mlir_npz/act_test_000000.npz \
  --test_reference act_mlir_000000.npz \
  --tolerance 0.90,0.50 \
  --model act_relu_mix_int8_bf16.cvimodel
```

## StarryOS 板端验证

将 mixed INT8/BF16 模型和测试输入复制到 TF 卡的 rootfs 中，然后在 SG2002 StarryOS 上使用 `/usr/bin/model_runner` 运行。

为了避免串口输入长文件名时被断行，板端使用了短链接：

```bash
ln -sf act_relu_mix_int8_bf16.cvimodel mix.cvimodel
ln -sf act_test_000000.npz a.npz
ln -sf act_test_000227.npz b.npz
```

对第一个测试样本连续运行 10 次：

```bash
/usr/bin/model_runner --model ./mix.cvimodel --input ./a.npz --count 10 --enable-timer --verbose 1
```

运行结果：

```text
Performance result: 10 runs take 681.513 ms, each run takes 68.151000 ms, fps 14.673299
Task exit with code: 0
```

对第二个测试样本连续运行 10 次：

```bash
/usr/bin/model_runner --model ./mix.cvimodel --input ./b.npz --count 10 --enable-timer --verbose 1
```

运行结果：

```text
Performance result: 10 runs take 681.393 ms, each run takes 68.139000 ms, fps 14.675883
Task exit with code: 0
```

带输出文件的单次运行命令如下：

```bash
/usr/bin/model_runner --model ./mix.cvimodel --input ./a.npz --output ./mix_a_out.npz --count 1 --enable-timer --verbose 1
/usr/bin/model_runner --model ./mix.cvimodel --input ./b.npz --output ./mix_b_out.npz --count 1 --enable-timer --verbose 1
```

带输出文件时的单次运行耗时：

```text
000000: 70.888 ms/run, fps 14.106760
000227: 70.938 ms/run, fps 14.096817
```

mixed INT8/BF16 模型在板端运行时的主要 ION carveout 申请为：

```text
ION alloc_buffer request: size=87399632, align=1, heap_type=Carveout
```

相比之前纯 BF16 模型约 `98,799,760 bytes` 的 ION carveout 申请，mixed INT8/BF16 模型的内存压力更低。

连续运行 10 次的性能结果：

```text
000000: 10 runs take 681.513 ms, each run takes 68.151 ms, fps 14.673299
000227: 10 runs take 681.393 ms, each run takes 68.139 ms, fps 14.675883
```

生成板端输出文件：

```bash
/usr/bin/model_runner --model ./mix.cvimodel --input ./a.npz --output ./mix_a_out.npz --count 1 --enable-timer --verbose 1
/usr/bin/model_runner --model ./mix.cvimodel --input ./b.npz --output ./mix_b_out.npz --count 1 --enable-timer --verbose 1
```

然后将板端输出拷回虚拟机，与 TPU-MLIR Docker 仿真输出进行对比。

对比结果：

```text
000000:
max_abs_diff = 0.0
mean_abs_diff = 0.0
cosine = 1.0

000227:
max_abs_diff = 0.0
mean_abs_diff = 0.0
cosine = 0.9999998807907104
```

这说明 StarryOS 板端输出与 TPU-MLIR 仿真输出一致。

## 结论

mixed INT8/BF16 的 ACT `cvimodel` 已经可以在 SG2002 StarryOS 上正常运行。

相比纯 BF16 模型，mixed INT8/BF16 模型大小从约 `96 MB` 降低到约 `85 MB`，运行速度从约 `84 ms/run` 提升到约 `68 ms/run`。

两个测试样本的板端输出与 Docker 仿真输出完成数值闭环验证。
