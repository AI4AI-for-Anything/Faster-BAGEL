# Faster-BAGEL

[English](README.md) | [简体中文](README.zh-CN.md)

**AI agent 优化的 BAGEL-7B-MoT 监督微调：在 8 张 Hopper GPU 上每步训练加速 1.82×。**

![Faster-BAGEL：8 张 Hopper GPU 上每步训练加速 1.82 倍](docs/performance-views/01-benchmark.svg)

| 实现 | 每步训练耗时（秒）↓ | 加速比 ↑ |
|---|---:|---:|
| 参考实现 | 1.4367 | 1.00× |
| **Faster-BAGEL** | **0.7873** | **1.82×** |

在 **8 张 Hopper GPU（80 GB）** 上用完整检查测得。负载是用官方配方对已发布的 BAGEL-7B-MoT 检查点（146.1 亿参数）做监督微调：FSDP `HYBRID_SHARD` 1×8，FP32 参数、FP32 AdamW 状态和 FP32 EMA，BF16 计算，每卡约 4,096 个打包 token，**不用 FP8**。计时用的是仓库里真实的训练循环：参考、候选、参考三个独立进程在同一台机器上依次运行，每次都用全新的编译缓存。候选随后以**统计等价**通过了检查的数值门，并不与参考逐位一致。精确比值为 **1.8249×**。目标是 3×，没有达到。

在 2,000 步的验证训练中，Faster-BAGEL 从启动到退出用时 **2,110–2,209 秒**，参考实现 **3,517–3,726 秒**，都含初始化和保存检查点。稳态每步约 **0.73–0.76 秒对 1.46–1.53 秒**。这几组跑在不同机器上，只作描述，不是配对测速。显存占用会上升：完整检查中每卡峰值分配 / 保留为 60.05 / 73.36 GiB，参考实现为 41.16 / 57.89 GiB。

## 快速开始

先准备[环境和公开数据](docs/training.md)，包括 900/100 的 VLM 划分，然后设置绝对路径。优化以环境变量开关的形式加在上游训练代码上；启动脚本会打开 M6 这组开关，并在 8 张 GPU 上运行官方 SFT 配方：

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

也可以先 `source scripts/configs/faster.env`，再用上游的 `torchrun ... train/pretrain_unified_navit.py` 命令，加上[记录在案的参数](docs/training.md#launch-and-baseline-corrections)。`REFERENCE=1` 会清掉所有开关。只在 **PyTorch 2.7.1 + CUDA 12.6（Triton 3.3.1）和 flash-attn 2.7.4.post1**、8 张 GPU 上测试过。所有开关默认关闭；都不设置时，代码走上游路径，只多两项不改变计算的修正：串行加载检查点、额外写一份指标日志。

## Agent 优化过程

图中每个点都是同一台机器上、由同一个冻结判卷做的一次完整检查，8 张 Hopper GPU。

![BAGEL 优化过程：1.21 倍到 1.82 倍](docs/performance-views/02-history.svg)

## 主要改动

- **显存与重算：** 上游在反向时把每个打了检查点的层都重算一遍，而 GPU 显存只用了一半左右。现在 LM、SigLIP 编码器和连接层都保留激活，不再重算。为了放得下，只重算便宜但占显存大的部分：RMSNorm 副本、SwiGLU 乘积，以及编译区域里的 MLP。
- **混合 Transformer（MoT）层：** 残差流按专家排好顺序，理解专家和生成专家直接读连续切片，不再在每个子模块里收集再散回 token。RMSNorm、SwiGLU 和 rotary 交给编译器融合；每个解码层在 FlexAttention 前后各编译成一段，FlexAttention 的分块大小也调过。每层前向的 CPU 时间大约减半。
- **FSDP 与负载均衡：**
  - 打开前向预取。FSDP 的梯度预除改成在分片上一次性后除：都是 2 的幂，数值完全一样。
  - SigLIP 参数从前向一直保留到反向，AdamW 换成融合版。
  - 在 LM 层前把打包好的文档在各卡之间重新分配、之后再还原，减轻最重那张卡的负担。
- **主机路径：** SigLIP 层改用手写反向，VAE 编码放到旁路流，去掉或推迟每步的主机同步。

![改动地图：显存、MoT 层、FSDP 与均衡、主机路径](docs/change-views/module-map.svg)

见[开关表](docs/training.md#switches)和[实测配置](scripts/configs/faster.env)。这些改动会改变浮点舍入和规约顺序，不是无损的。

## 验证

**数值检查。** 最终配置通过了[冻结的判卷](docs/judge.md)。判卷直接在 7B 负载上搭建和标定，没有用小模型代替。

- **精确项**：结构、输入 batch、随机数指纹、调用次数和学习率，全部没有失败。
- **逐位比对**：记录的 5,696 个张量中只有 1,038 个逐位相同，所以由统计门判定。
- **统计门**：三个种子各跑一条 260 步的配对轨迹，14 个指标覆盖 CE、MSE、梯度范数，以及 ViT、LM 和其余参数的位移与 EMA 位移。14 项全部在 δ 以内，δ 是参考实现自身种子间标准差的两倍，由六个参考种子冻结而来。

判卷在验收中拒掉了所有改变语义的改动，但余量偏薄：
- **LR×1.02 负对照**：只被 EMA 位移抓到，六个种子下为 +1.43 δ 和 +1.16 δ。
- **M7**：后来的候选，测速 1.8972×，因 LM 位移偏离 −3.0 δ 被判为不正确而拒掉。

**长训练。** Faster-BAGEL 和参考实现各用四个种子（全局种子 4396–4399，数据种子 42）跑满完整的 2,000 步配方。每组最终的模型和 EMA，都用参考评测器在 100 条留出 VLM 样本上打分。这些样本的图片与训练不重叠，共 8,407 个文本 token。所有运行都没有出现非有限值。

![每种实现四个种子：训练曲线、合并后的窗口差和留出集 CE](docs/performance-views/04-validation.svg)

| 分析 | 训练 CE | 训练 MSE | 梯度范数对数 | 留出集模型 / EMA | 结论 |
|---|---:|---:|---:|---|---|
| 第一轮：两个种子，每个窗口要求 \|E\| ≤ \|Y\| | 9 / 20 | 8 / 20 | 5 / 20 | 都不过 | **不一致** |
| 第二轮：四个种子，\|D\| ≤ 2σ<sub>参考</sub>，判据在运行前定好 | 19 / 20 | 20 / 20 | 19 / 20 | 都通过 | **一致** |

- **第一轮**：用参考实现两个种子之差当尺子。两个种子的数据顺序相同，这把尺子经常极小（某个 CE 窗口里只有 4×10⁻⁶），结果不通过。这个结论原样保留。
- **第二轮**：判据在两个新种子开跑前就写定。D = Faster-BAGEL 均值 − 参考均值，δ = 参考四个种子样本标准差的两倍。每项训练指标最多允许两个窗口不过，且最后一个窗口必须通过。
  - 最后一个窗口全部通过。
  - 20 个窗口上的符号检验 p 值为 0.50（CE）、0.041（MSE）、0.26（梯度范数），都没有触发 p < 0.01 的偏差标记。其中 MSE 有 15 个窗口是 Faster-BAGEL 略低。
  - 留出集模型 CE 为 1.3423 对 1.3294（D = +0.0129，δ = 0.0683），EMA CE 为 0.74328 对 0.74489（D = −0.0016，δ = 0.0019）。

**注意。** 文生图和编辑的样本分别是同一张图、同一条四图编辑链在重复，它们的损失只反映对固定样本的拟合。只有留出集 VLM CE 能说明泛化，生成质量没有评测。两种实现都出现过大于 10 的梯度范数尖峰：每组的最大值，参考实现为 25–86，Faster-BAGEL 为 60–91。四个种子是在上述判据下的证据，不能证明两者的训练分布完全相同。

见[验证记录](docs/validation/)和[复现脚本](scripts/validate/README.md)。

## 为什么停在 1.82×

3× 意味着每步不超过 479 毫秒。最终配置最新的 profile 里，单卡每步光是 GPU 计算 kernel 就要 634 毫秒，还没算通信。另外几个限制：

- 打包方式由配方固定，最重那张卡的 token 数是平均的 1.24 倍；
- 前向仍受 CPU 发射 kernel 的限制；
- GEMM 只跑到 BF16 峰值的约 51%。

更早的估计认为，用这些 kernel，即使等待和负载不均全部消失，上限也只有约 2.1×。FP8 不在允许范围内。第 34–45 轮都没能再带来认证过的提速，所有者在 M6 结束了优化。

![最终配置的时间预算](docs/performance-views/03-time-budget.svg)

见[放弃的方向](docs/training.md#rejected-directions)和[判卷记录](docs/judge.md)。

## AI4AI-for-Anything 的其他项目

[Faster-YOCO](https://github.com/AI4AI-for-Anything/Faster-YOCO) · [Faster-JiT](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Faster-RAE](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Faster-LLaVA](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Faster-DMD](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Faster-SelfForcing](https://github.com/AI4AI-for-Anything/Faster-SelfForcing)

## 上游 BAGEL 文档

基于 [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) 的 `a2fa77d`，即论文 [Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683) 的代码，[Apache-2.0 许可证](LICENSE)。`upstream` 分支保存未改动的代码。原始 README 保存在 [README.upstream.md](README.upstream.md)，也附在下面。

<details>
<summary>原始 BAGEL README：模型、安装、训练、评测</summary>

<p align="center">
  <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="480"/>
</p>

<p align="center">
  <a href="https://bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white"
      alt="BAGEL Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red"
      alt="BAGEL Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
    <img 
        src="https://img.shields.io/badge/BAGEL-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="BAGEL Model"
    />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"
      alt="BAGEL Demo"
    />
  </a>
  <a href="https://huggingface.co/spaces/ByteDance-Seed/BAGEL">
    <img 
        src="https://img.shields.io/badge/BAGEL-Space-orange?logo=huggingface&logoColor=yellow" 
        alt="BAGEL Model"
    />
  </a>
  <a href="https://discord.gg/eXQNFhWe">
    <img
      src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple"
      alt="BAGEL Discord"
    />
  </a>
  <a href="mailto:bagel@bytedance.com">
    <img
      src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red"
      alt="BAGEL Email"
    />
  </a>
</p>

# Unified Model for Multimodal Understanding and Generation
> [Chaorui Deng*](https://scholar.google.com/citations?hl=en&user=k0TWfBoAAAAJ), [Deyao Zhu*](https://tsutikgiau.github.io/), [Kunchang Li*](https://andy1621.github.io/), [Chenhui Gou*](https://www.linkedin.com/in/chenhui-gou-9201081a1/?originalSubdomain=au), [Feng Li*](https://fengli-ust.github.io/), [Zeyu Wang](https://zw615.github.io/), Shu Zhong, [Weihao Yu](https://whyu.me/), [Xiaonan Nie](https://codecaution.github.io/), [Ziang Song](https://www.linkedin.com/in/ziang-song-43b0ab8a/), Guang Shi :email: , [Haoqi Fan* :tophat: ](https://haoqifan.github.io/)
>
> contact: shiguang.sg@bytedance.com
> 
> We present **BAGEL**, an open‑source multimodal foundation model with 7B active parameters (14B total) trained on large‑scale interleaved multimodal data. BAGEL outperforms the current top‑tier open‑source VLMs like Qwen2.5-VL and InternVL-2.5 on standard multimodal understanding leaderboards, and delivers text‑to‑image quality that is competitive with strong specialist generators such as SD3.
Moreover, BAGEL demonstrates superior qualitative results in classical image‑editing scenarios than the leading open-source models. More importantly, it extends to free-form visual manipulation, multiview synthesis, and world navigation, capabilities that constitute "world-modeling" tasks beyond the scope of previous image-editing models.
The figure below showcases BAGEL's qualitative performance.

<p align="center"><img src="assets/teaser.webp" width="95%"></p>


<!-- ## 🧠 Method
BAGEL adopts a Mixture-of-Transformer-Experts (MoT) architecture to maximize the model’s capacity to learn from richly diverse multimodal information. Following the same principle of capacity maximization, it utilizes two separate encoders to capture pixel-level and semantic-level features of an image. The overall framework follows a Next Group of Token Prediction paradigm, where the model is trained to predict the next group of language or visual tokens as a compression target.

BAGEL scales MoT’s capacity through Pre-training, Continued Training, and Supervised Finetuning on trillions of interleaved multimodal tokens spanning language, image, video, and web data. It surpasses open models on standard understanding and generation benchmarks and demonstrates advanced in-context multimodal abilities like free-form image editing, future frame prediction, 3D manipulation, world navigation, and sequential reasoning.

<p align="center"><img src="assets/arch.png" width="95%"></p>


## 🌱 Emerging Properties
<p align="center"><img src="assets/emerging_curves.png" width="95%"></p>

As we scale up BAGEL’s pretraining with more multimodal tokens, we observe consistent performance gains across understanding, generation, and editing tasks. Different capabilities emerge at distinct training stages—multimodal understanding and generation appear early, followed by basic editing, while complex, intelligent editing emerges later. This staged progression suggests an emergent pattern, where advanced multimodal reasoning builds on well-formed foundational skills. Ablation studies further show that combining VAE and ViT features significantly improves intelligent editing, underscoring the importance of visual-semantic context in enabling complex multimodal reasoning and further supporting its role in the emergence of advanced capabilities. -->

## 📢 News

We sincerely thank all contributors from the open community for their valuable support.

- **June 15, 2025:** We have updated and fixed the evaluation results for [KRIS-Bench](https://github.com/mercurystraw/Kris_Bench) and [RISEBench](https://github.com/PhoenixZ810/RISEBench). **Our model, BAGEL, demonstrates performance comparable to Gemini 2.0 on these reasoning benchmarks.** We have also released the evaluation code for both KRIS-Bench and RISEBench, along with [ImgEdit-Bench](https://github.com/PKU-YuanGroup/ImgEdit). For further details, please refer to [EVAL](./EVAL.md).
- **Jun 5, 2025:** Thanks to [@davideuler](https://github.com/davideuler) for contributing the [Dockerfile with prebuilt flash_attn](https://github.com/ByteDance-Seed/Bagel/issues/125).
- **May 30, 2025:** Many thanks to [@prartio](https://github.com/prartio) for contributing the [Windows 11 installation guideline](https://github.com/ByteDance-Seed/Bagel/issues/92), and to [@gluttony-10](https://github.com/gluttony-10) for his work on the [inference of quantization](https://github.com/ByteDance-Seed/Bagel/pull/88).
- **May 29, 2025:** Special thanks to [@jnc-nj](https://github.com/jnc-nj) for contributing the [Dockerfile](https://github.com/ByteDance-Seed/Bagel/issues/75).
- **May 26, 2025:** Thanks to [@neverbiasu](https://github.com/neverbiasu) for contributing [ComfyUI](https://github.com/neverbiasu/ComfyUI-BAGEL).
- **May 25, 2025:** Special thanks to [@LeanModels](https://github.com/LeanModels) for providing the [DF11-compressed version](https://huggingface.co/DFloat11/BAGEL-7B-MoT-DF11), and to [@Gapeleon](https://huggingface.co/Gapeleon) for the [INT8-compressed version](https://huggingface.co/Gapeleon/bytedance_BAGEL-7B-MoT-INT8). We also appreciate [@gluttony-10](https://github.com/gluttony-10) for contributions to the [Windows package](https://github.com/ByteDance-Seed/Bagel/issues/51).
- **May 24, 2025:** Together with [@wangwei1237](https://github.com/wangwei1237), [@gluttony-10](https://github.com/gluttony-10), and [@KingNish24](https://github.com/KingNish24), we built a Gradio [app](app.py) and launched a [Hugging Face Space](https://huggingface.co/spaces/ByteDance-Seed/BAGEL).
- **May 23, 2025:** We have provided a training guideline in [TRAIN](./TRAIN.md).
- **May 20, 2025:** We released the official [website](https://bagel-ai.org/), [demo](https://demo.bagel-ai.org/), [model](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT), and [report](https://arxiv.org/abs/2505.14683) for BAGEL.


## 📮 Notice
**Call for Bad Cases:** If you have encountered any cases where the model performs poorly, we would greatly appreciate it if you could share them in the [issue#11](https://github.com/ByteDance-Seed/Bagel/issues/11) or [Discord](https://discord.gg/Z836xxzy).

**About Inference Hyperparameters:**
- **`cfg_text_scale`:** Controls how strongly the model follows the text prompt. `1.0` disables text guidance. Typical range: `4.0–8.0`.
- **`cfg_image_scale`:** Controls how much the model preserves input image details. `1.0` disables image guidance. Typical range: `1.0–2.0`.
- **`cfg_interval`:** Fraction of denoising steps where CFG is applied. Later steps can skip CFG to reduce computation. Typical: `[0.4, 1.0]`.
- **`timestep_shift`:** Shifts the distribution of denoising steps. Higher values allocate more steps at the start (affects layout); lower values allocate more at the end (improves details).
- **`num_timesteps`:** Total denoising steps. Typical: `50`.
- **`cfg_renorm_min`:** Minimum value for CFG-Renorm. `1.0` disables renorm. Typical: `0`.
- **`cfg_renorm_type`:** CFG-Renorm method:  
  - `global`: Normalize over all tokens and channels (default for T2I).
  - `channel`: Normalize across channels for each token.
  - `text_channel`: Like `channel`, but only applies to text condition (good for editing, may cause blur).
- **If edited images appear blurry, try `global` CFG-Renorm, decrease `cfg_renorm_min` or decrease `cfg_scale`.**


## 🔥 Quick Start

1️⃣  Set up environment
```bash
git clone https://github.com/bytedance-seed/BAGEL.git
cd BAGEL
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

2️⃣  Download pretrained checkpoint
```python
from huggingface_hub import snapshot_download

save_dir = "models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

```

3️⃣ Use Gradio WebUI to start playing with BAGEL!
```bash
# For 32GB+ VRAM GPU or multi GPUs.
python app.py
```

```bash
# For 12~32GB VRAM GPU, recommend using NF4 quantization. And use Chinese interface.
python app.py --mode 2 --zh
```

```bash
# For 22~32GB VRAM GPU, not recommended to use INT8 quantization.
python app.py  --mode 3
```

## 🔥 Train & Eval

### Train

```bash
bash scripts/train.sh
```

You can replace the variables in the script with your own before running. 
See [TRAIN](TRAIN.md) for more details.

### Eval
We provide the scripts for evaluating VLM, T2I and Editing benchmarks. 
Please See [EVAL](EVAL.md) for more details.


## 📊 Benchmarks

### 1. Visual Understanding

| Model | MME | MMBench |   MMMU | MM-Vet | MathVista |
| ------------------- | ----------: | ----------: | -------: | -------: | ----------: |
| Janus-Pro-7B        | -  |     79.2 |     41.0 |     50.0 |           – |
| Qwen2.5-VL-7B      | 2347    |   83.5 | **58.6** |     67.1 |           68.2 |
| **BAGEL**    | **2388**  |  **85.0** |     55.3 | **67.2** |    **73.1** |

### 2. Text-to-Image Generation

| Model        | GenEval | WISE |
| ------------ | --------- | --------- |
| Janus-Pro-7B | 0.80      | 0.35 | 
| SD3-Medium   | 0.74      | - |
| FLUX-1-dev   | 0.82      | 0.50 |
| **BAGEL**    | 0.82  | 0.52  |
| **BAGEL + Rewritter/CoT**    | **0.88**  | **0.70** |

### 3. Image Editing

| Model         | GEdit-Bench-EN (SC) | GEdit-Bench-EN (PQ) | GEdit-Bench-EN (O) | IntelligentBench | KISE-Bench | RISEBench |
| ------------- | ---------------------: | ---------------------: | -------------------: | ------------------: | ------------: | ------------: | 
| Step1X-Edit   | 🥉7.09                | 🥉6.76                | 🥈6.70            | 14.9               |  43.29   |  1.9  |
| Gemini 2.0    | 6.73                  | 6.61                  | 6.32                | 🥈57.6             | 🥈62.41   |  🥈13.3  |
| GPT-4o        | 🥇7.85              | 🥇7.62              | 🥇7.53            | 🥇78.9           | 🥇80.09   |  🥇28.9  |
| **BAGEL**     | 🥈7.36                | 🥈6.83                | 🥉6.52                | 44.0               |  56.21   |  6.1 |
| **BAGEL+CoT** | –                     | –                     | –                   | 🥉55.3             |  🥉60.18   |  🥉11.9 |




## ✍️ Citation

```bibtex
@article{deng2025bagel,
  title   = {Emerging Properties in Unified Multimodal Pretraining},
  author  = {Deng, Chaorui and Zhu, Deyao and Li, Kunchang and Gou, Chenhui and Li, Feng and Wang, Zeyu and Zhong, Shu and Yu, Weihao and Nie, Xiaonan and Song, Ziang and Shi, Guang and Fan, Haoqi},
  journal = {arXiv preprint arXiv:2505.14683},
  year    = {2025}
}
```


## 📜 License
BAGEL is licensed under the Apache 2.0.

</details>
