# Faster-BAGEL

[English](README.md) | [简体中文](README.zh-CN.md)

**AI agent 优化的 BAGEL-7B-MoT 监督微调：在 8 × 80 GB Hopper GPUs 上，训练步加速 1.8249×。**

![基准测试](docs/performance-views/01-benchmark.svg)

| Implementation | Seconds / step ↓ | Speedup ↑ |
|---|---:|---:|
| Reference | 1.4367 | 1.0000× |
| **Faster-BAGEL (M6)** | **0.7873** | **1.8249×** |

使用官方发布 checkpoint 的 SFT 配方，总参数 146.1 亿，FSDP HYBRID_SHARD 1×8，参数、AdamW 状态和 EMA 为 FP32，计算为 BF16。完整检查在独立进程中按原版→优化版→原版测速，使用冷编译缓存，再检查数值。M6 通过 N1 统计等价检查，并非逐位一致；3× 目标未达到。

## 快速开始

先阅读[训练说明](docs/training.md)，准备公开权重、示例数据和 900/100 的 VLM 训练/留出划分。设置绝对路径：

```bash
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export BAGEL_DATA_ROOT=/path/to/bagel_example
export BAGEL_TRAIN_JSONL=/path/to/train.jsonl
export BAGEL_EDIT_META=/path/to/edit_meta.json
export OUTPUT_DIR=/path/to/results
export CHECKPOINT_DIR=/path/to/local/checkpoints
export CACHE_DIR=/path/to/local/fresh-cache
export GPU_LOCK=/path/to/shared-gpu.lock
bash scripts/train_faster_bagel_sft.sh
```

启动脚本显式启用 M6；代码的所有优化开关默认关闭。`REFERENCE=1` 清除开关，保留双方共同的串行权重加载和记录功能。已验证的环境为 PyTorch 2.7.1+cu126、Triton 3.3.1、flash-attn 2.7.4.post1。便携脚本完成静态和命令构建检查，没有另起训练。

## 主要改动

选择性重计算、MoT 残差流排列、编译的逐元素与解码器区域、FlexAttention 调优、融合 AdamW；FSDP 预取/后除法、ViT 参数保留与显式反向、异步 VAE 和延迟同步；最后通过可逆的整文档 LM 均衡降低卡间负载差。见[开关表](docs/training.md#switches)。完整检查每卡峰值分配/保留内存为 60.05/73.36 GiB，原版为 41.16/57.89 GiB。

## Agent 优化过程

![优化历史](docs/performance-views/02-history.svg)

M1–M6 均通过完整检查。M7 虽然测得 1.8972×，但数值结论为 **INCORRECT**，已拒绝。里程碑是累计配置，不是每个开关的独立消融。见[检查规则与控制组](docs/judge.md)。

## 验证

两轮结论同时保留。每个实现以 4396–4399 四个全局种子各训练 2000 步，data_seed 固定为 42。八次训练全部完成；最终 model 和 EMA 由同一个参考评测实现，在与训练图片内容不重叠的 100 行 VLM 留出集上评分，共 8407 个监督 token，关闭 dropout。全部记录有限，没有重跑。

| 分析 | CE 一致窗口 | MSE 一致窗口 | log-grad 一致窗口 | model / EMA 留出 CE | 结论 |
|---|---:|---:|---:|---|---|
| 第一轮，两个种子 | 9/20 | 8/20 | 5/20 | 均失败 | **NOT CONSISTENT** |
| 第二轮，四个种子合并 | 19/20 | 20/20 | 19/20 | 均通过 | **CONSISTENT** |

第一轮用原版两种子之差作为尺度，最终 CE 窗口未通过；原结论保持不变。第二轮规则在新增运行前固定：δ=两倍原版四种子的样本标准差，D=优化版均值−原版均值；|D|≤δ 为一致。训练指标最多允许 2/20 窗口失败且末窗必须通过。末窗全部通过；双侧符号检验 p 分别为 0.5034、0.04139、0.2632，均未触发 p<0.01 偏差标记。配对 t 检验只作描述。

合并 model CE 为原版 1.3293698、M6 1.3422934，差 +0.0129236，δ=0.0683410；EMA CE 为 0.74488545、0.74327525，差 −0.00161020，δ=0.00185845。四种子通过指定判据并不能证明训练分布完全相同。

![两轮验证](docs/performance-views/04-validation.svg)

完整[第一轮](docs/validation/verdict.md)和[第二轮](docs/validation/round2/verdict.md)结果已脱敏；16000 步指标及 1600 行评测分数均可[在 CPU 上复算](scripts/validate/README.md)。验证训练的启动至退出时间：原版 3517–3726 秒，M6 2110–2209 秒，含初始化及保存。运行位于不同机器，这些时间不是同机配对测速的加速比。

## 限制

t2i 数据重复同一张图片，编辑数据重复一条四图链；相应训练 loss 只能说明对重复样例的拟合。泛化证据仅来自不重叠的 VLM 留出 CE，未进行生成质量基准评测。双方均出现过梯度范数 >10 的尖峰，报告逐次列出。

较早的 profile 给出现有计算核约 **2.1×** 的估算上限：最重 rank 与均值 token 数比 1.24，forward 受 CPU 开销影响，GEMM 约达 BF16 峰值的 51%。最新 M6 profile 中计算并集为 633.8 ms、暴露通信为 220.6 ms；profile 含额外开销，不能代替正式测速。该估计不是所有未来算法的数学极限，3× 目标未达到。

![时间构成](docs/performance-views/03-time-budget.svg)

基于 [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) `a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f`；`upstream` 分支保留原始树，原 README 原样保存在 [README.upstream.md](README.upstream.md)。[Apache-2.0](LICENSE)。
