# vLLM 与 ModelScope SWIFT 

---

## 目录

- [0. 学完能做什么](#0-学完能做什么)
- [1. 先弄清 vLLM、SWIFT 和 Hugging Face 的关系](#1-先弄清-vllmswift-和-hugging-face-的关系)
- [2. 从零认识模型、推理、训练和 LoRA](#2-从零认识模型推理训练和-lora)
- [3. 准备 Linux、显卡和 Python 环境](#3-准备-linux显卡和-python-环境)
- [4. 下载并检查第一个模型](#4-下载并检查第一个模型)
- [5. vLLM 入门：离线批量推理](#5-vllm-入门离线批量推理)
- [6. vLLM 实战：启动 OpenAI 兼容服务](#6-vllm-实战启动-openai-兼容服务)
- [7. vLLM 参数、显存与多卡](#7-vllm-参数显存与多卡)
- [8. SWIFT 入门：安装和认识命令](#8-swift-入门安装和认识命令)
- [9. 准备自己的 SFT 数据集](#9-准备自己的-sft-数据集)
- [10. 用 SWIFT 做 LoRA 微调](#10-用-swift-做-lora-微调)
- [11. 用 SWIFT 测试 LoRA 效果](#11-用-swift-测试-lora-效果)
- [12. 合并 LoRA，并交给 vLLM 部署](#12-合并-lora并交给-vllm-部署)
- [13. 不合并 LoRA 的部署方式](#13-不合并-lora-的部署方式)
- [14. SWIFT 的 GRPO 强化学习入口](#14-swift-的-grpo-强化学习入口)
- [15. 一条完整工作流](#15-一条完整工作流)
- [16. 常见报错排查](#16-常见报错排查)
- [17. 练习路线与验收清单](#17-练习路线与验收清单)
- [18. 常用命令速查](#18-常用命令速查)
- [19. 术语表](#19-术语表)

---

# 0. 学完能做什么

完成本文后，你应该能独立完成下面这条链路：

```text
选择基础模型
   ↓
用 vLLM 运行原始模型，确认模型能回答
   ↓
把自己的问答整理成 messages 格式的 JSONL
   ↓
用 SWIFT 对模型做 LoRA 监督微调（SFT）
   ↓
用 SWIFT 对比微调前后的回答
   ↓
合并 LoRA 权重，或直接加载 LoRA
   ↓
用 vLLM / SWIFT Deploy 启动 OpenAI 兼容 API
   ↓
让 Python 程序、网页或其他应用调用模型
```

本文使用 `Qwen/Qwen3-4B` 作为示例。它只是示例，不是硬性要求。如果显存较小，可换成当前版本可用的 0.6B、1.5B 或 1.7B 级别指令模型；如果显存充足，也可以换成 7B/8B 模型。

所有示例中的模型 ID、路径和端口都可以修改。看到以下形式时，要理解它是“变量”，不是必须原样照抄：

```bash
Qwen/Qwen3-4B                 # Hugging Face / ModelScope 模型 ID
/root/autodl-tmp/models/...   # Linux 本地路径
8000                          # API 端口
```

---

# 1. 先弄清 vLLM、SWIFT 和 Hugging Face 的关系

## 1.1 三者分别解决什么问题

| 工具 | 主要用途 | 最适合的场景 |
|---|---|---|
| Hugging Face Transformers | 模型加载、训练和基础推理的通用库 | 学习模型代码、灵活实验、训练 |
| vLLM | 高吞吐大模型推理与服务 | 批量生成、多人并发、部署 API |
| ModelScope SWIFT（ms-swift） | 大模型训练、微调、推理、评测和部署工具箱 | 用较少代码做 SFT、LoRA、DPO、GRPO 等 |

最简单的记法：

- **SWIFT 负责“教模型”**：用数据微调模型。
- **vLLM 负责“让模型高效工作”**：把训练好的模型快速提供给用户或程序调用。
- **Transformers 是共同的基础设施之一**：模型结构、Tokenizer、权重格式经常都来自它的生态。

SWIFT 也能推理和部署，SWIFT 的部分推理/rollout 场景还能使用 vLLM 作为后端。因此两者不是互相替代，而是经常组合使用。

## 1.2 为什么不能只用 `transformers.generate()`

单人测试时，Transformers 已经够用。但服务有很多并发请求时，传统批处理会遇到：

- 不同请求长度不同，短请求可能等待长请求；
- KV Cache 显存利用率不高；
- 请求动态进入和退出时不容易持续把 GPU 填满；
- 自己写排队、流式输出、API、并发控制很麻烦。

vLLM 的核心价值是 PagedAttention、连续批处理和成熟的服务接口。它主要提高的是**总体吞吐量**，不保证单个短请求的首字延迟一定大幅下降。

## 1.3 为什么初学者适合用 LoRA

完整更新 4B 模型的全部参数，不仅要存模型权重，还要存梯度、优化器状态和激活值，显存需求很高。LoRA 冻结原模型，只训练插入到部分线性层的小矩阵：

- 训练参数少；
- 显存需求低很多；
- 训练产物通常只有几十 MB 到几百 MB；
- 同一个基础模型可以挂载多个不同任务的 LoRA。

所以本文的第一条训练路线是 **SFT + LoRA**。

---

# 2. 从零认识模型、推理、训练和 LoRA

## 2.1 一个大模型目录里有什么

下载一个 Hugging Face 格式的模型后，通常能看到：

| 文件 | 作用 |
|---|---|
| `config.json` | 层数、隐藏维度、注意力头等模型结构配置 |
| `tokenizer.json` / `tokenizer_config.json` | 把文字和 token ID 相互转换 |
| `*.safetensors` | 模型权重，通常是最大的文件 |
| `model.safetensors.index.json` | 多个权重分片的索引 |
| `generation_config.json` | 默认生成配置 |
| `chat_template`（常在 tokenizer 配置中） | 把 system/user/assistant 消息拼成模型认识的文本 |

只复制一个 `.safetensors` 文件通常不够。部署时应保留完整模型目录。

## 2.2 Tokenizer 是什么

模型不直接读取汉字或英文单词，而是读取整数 ID。Tokenizer 做两件事：

```text
“你好” → [108386]       编码 encode
[108386] → “你好”       解码 decode
```

不同模型的词表不同。不能随意把 A 模型的 Tokenizer 配给 B 模型。

## 2.3 Chat Template 是什么

对话模型不仅需要内容，还要知道谁在说话。程序里的消息可能是：

```json
[
  {"role": "system", "content": "你是一个严谨的助手。"},
  {"role": "user", "content": "什么是 LoRA？"}
]
```

Chat Template 会把它转换成模型训练时见过的特殊格式。直接手写 `用户：... 助手：...` 可能与训练模板不一致，导致效果变差。因此对话模型应优先使用它自带的 Chat Template。

## 2.4 推理、SFT 和强化学习的区别

| 阶段 | 模型参数是否更新 | 输入 | 反馈 |
|---|---:|---|---|
| 推理 Inference | 否 | 用户问题 | 模型生成回答 |
| SFT 监督微调 | 是 | 问题 + 标准答案 | 用标准答案计算损失 |
| 偏好/RL 训练 | 是 | 问题 + 偏好或奖励 | 根据偏好/奖励优化 |

第一次学习时，先跑通推理，再做 SFT，最后才碰 GRPO。不要一开始就把所有问题叠在一起。

## 2.5 训练产物是什么

LoRA 训练后常见的目录结构是：

```text
output/
└── v0-20260718-120000/
    ├── checkpoint-50/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── ...
    └── checkpoint-100/
        └── ...
```

`checkpoint-100` 表示训练到第 100 个优化步骤时保存的检查点。它通常只包含 LoRA 适配器，不等于完整基础模型。使用时必须同时知道基础模型是谁。

---

# 3. 准备 Linux、显卡和 Python 环境

## 3.1 系统要求

最省事的组合是：

- Ubuntu 22.04；
- NVIDIA 驱动正常；
- Python 3.10 或 3.11；
- 一张 CUDA 可用的 NVIDIA GPU；
- 足够的磁盘空间，建议至少预留模型大小的 2～3 倍。

vLLM 的主流 CUDA 使用方式面向 Linux。Windows 用户建议使用 WSL2，而不是直接在原生 PowerShell 环境安装 CUDA 版 vLLM。

## 3.2 Windows 用户先分清两个路径世界

Windows 路径：

```text
D:\学习\AR科研\week3
```

在 WSL2 中通常对应：

```text
/mnt/d/学习/AR科研/week3
```

训练和模型读取更适合放在 WSL2 的 Linux 文件系统，例如 `~/models`、`~/work`，因为从 `/mnt/d` 大量读取小文件有时较慢。

## 3.3 第一步：检查 GPU 和驱动

在 Linux 终端运行：

```bash
nvidia-smi
```

成功时会看到 GPU 型号、Driver Version、显存占用和进程。再运行：

```bash
which python
python --version
```

如果 `nvidia-smi` 不存在或报错，先修复驱动/WSL GPU 映射；这不是安装 Python 包可以解决的问题。

## 3.4 安装 Miniconda（已有可跳过）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
```

安装程序询问是否初始化 shell 时选 `yes`。如果 `conda` 仍找不到，可以重新打开终端。

## 3.5 建立两个环境更容易排错

初学阶段建议把推理和训练分开：

```bash
conda create -n vllm-env python=3.11 -y
conda create -n swift-env python=3.11 -y
```

原因是 vLLM、PyTorch、FlashAttention 和训练工具对 CUDA/PyTorch 版本有组合要求。分环境可以避免为了安装一个包把另一个已工作的环境升级坏。

后面需要 SWIFT 调用 vLLM 后端时，再按当前 SWIFT 文档把兼容的 vLLM 装进 `swift-env`；不要盲目在一个能工作的环境里同时执行多个 `pip install -U`。

## 3.6 建立工作目录

```bash
mkdir -p ~/llm-lab/{models,data,output,scripts}
cd ~/llm-lab
```

以后约定：

```text
~/llm-lab/models    完整基础模型或合并后的模型
~/llm-lab/data      JSONL 数据集
~/llm-lab/output    SWIFT 检查点和日志
~/llm-lab/scripts   Python 测试程序
```

## 3.7 记录环境，养成可复现习惯

每次实验至少记录：

```bash
nvidia-smi
python --version
pip list | grep -E "torch|vllm|swift|transformers|modelscope"
```

发生问题时，“我装的是哪个版本”比“我好像装过”有用得多。

---

# 4. 下载并检查第一个模型

有两种常用方式：直接使用在线模型 ID，让框架自动下载；或提前下载到本地目录。

## 4.1 方式 A：直接使用模型 ID

后面的命令写：

```bash
--model Qwen/Qwen3-4B
```

第一次运行会自动下载，后面从缓存读取。优点是简单，缺点是缓存位置不直观，网络中断时也不方便控制。

## 4.2 方式 B：用 ModelScope 下载到明确目录

进入任一 Python 环境：

```bash
conda activate swift-env
pip install -U modelscope
```

下载：

```bash
modelscope download --model Qwen/Qwen3-4B --local_dir ~/llm-lab/models/Qwen3-4B
```

如果当前 `modelscope` 版本的命令形式不同，先查看：

```bash
modelscope download --help
```

也可以用 Python：

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-4B",
    local_dir="/home/你的用户名/llm-lab/models/Qwen3-4B",
)
print(model_dir)
```

不要在 Python 字符串里写 `~/llm-lab/...` 并期待它总能自动展开。最稳妥的是写完整绝对路径，或用 `Path.home()` 组合路径。

## 4.3 检查目录是否完整

```bash
ls -lh ~/llm-lab/models/Qwen3-4B
du -sh ~/llm-lab/models/Qwen3-4B
```

至少应看到配置、Tokenizer 和权重文件。若目录只有几 KB，通常是下载未完成或拿到了链接/元数据而非完整权重。

## 4.4 选择模型时看三件事

1. **是不是生成式语言模型**：vLLM 本文示例面向 Causal LM。
2. **是不是 Instruct/Chat 模型**：要对话就优先选指令版。
3. **显存是否够**：BF16 权重粗略占用约为参数量 × 2 字节，此外还需要 KV Cache 和框架开销。

例如 4B BF16 权重理论值约 8 GB，但实际运行不能只准备 8 GB，因为还要给 KV Cache、CUDA Kernel 和临时张量留空间。

---

# 5. vLLM 入门：离线批量推理

“离线推理”是指 Python 程序直接加载模型并生成，不启动网络 API。它适合批量处理文件、做评测和理解 vLLM 对象。

## 5.1 安装 vLLM

```bash
conda activate vllm-env
python -m pip install -U pip
pip install vllm
```

安装后检查：

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

最后一个值必须是 `True`。如果是 `False`，说明当前 PyTorch 没识别到 CUDA，不要继续调 vLLM 参数。

## 5.2 最小可运行程序

新建 `~/llm-lab/scripts/vllm_offline.py`：

```python
from vllm import LLM, SamplingParams

MODEL = "/home/你的用户名/llm-lab/models/Qwen3-4B"

llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
)

conversations = [
    [{"role": "user", "content": "请用三句话解释什么是 LoRA。"}],
    [{"role": "user", "content": "vLLM 主要解决什么问题？"}],
]

outputs = llm.chat(
    conversations,
    sampling_params=sampling_params,
    chat_template_kwargs={"enable_thinking": False},
)

for index, output in enumerate(outputs, start=1):
    print(f"\n===== 回答 {index} =====")
    print(output.outputs[0].text)
```

把 `MODEL` 改成你机器上的真实路径，然后运行：

```bash
python ~/llm-lab/scripts/vllm_offline.py
```

如果当前模型不是带 `enable_thinking` 开关的 Qwen 模型，删掉 `chat_template_kwargs` 即可。若当前 vLLM 版本的 `chat()` 参数签名有差异，运行下面命令确认：

```bash
python -c "from vllm import LLM; help(LLM.chat)"
```

## 5.3 程序逐行解释

`LLM(...)` 创建推理引擎并加载模型：

- `model`：在线模型 ID 或完整本地目录；
- `dtype`：权重/计算精度，支持情况取决于 GPU；
- `max_model_len`：一次请求允许的最大上下文长度；
- `gpu_memory_utilization`：vLLM 可用于权重、KV Cache 等的显存比例目标。

`SamplingParams(...)` 控制生成：

- `temperature=0.0`：近似确定性选择，适合测试、抽取、数学题；
- `max_tokens=256`：最多生成 256 个新 token，不包含输入 token；
- 还可设置 `top_p`、`top_k`、`stop`、`repetition_penalty` 等。

输出对象不是一个普通字符串：

```text
outputs                         所有请求结果
outputs[0]                      第 1 个请求
outputs[0].outputs[0]           第 1 个候选回答
outputs[0].outputs[0].text      回答文本
outputs[0].outputs[0].token_ids 生成 token ID
```

## 5.4 `generate()` 和 `chat()` 的区别

- `chat()` 接受带 `role` 的消息，并帮助应用 Chat Template；
- `generate()` 接受已经处理好的 prompt 字符串，控制更直接；
- 做普通对话优先 `chat()`；做特殊模板、补全或精确评测时可用 `generate()`。

## 5.5 如何判断跑通

满足以下条件即可：

- 终端没有 CUDA OOM；
- 模型权重加载完成；
- 两个问题都生成了非空回答；
- `nvidia-smi` 能看到 Python 进程占用显存。

第一次启动较慢是正常的，因为包含下载、加载权重、显存分析和可能的 Kernel 编译。不要把首次启动时间当成生成速度。

---

# 6. vLLM 实战：启动 OpenAI 兼容服务

## 6.1 什么叫 OpenAI 兼容 API

它表示请求路径和 JSON 结构与常见 OpenAI SDK 接口兼容，例如：

```text
POST /v1/chat/completions
POST /v1/completions
GET  /v1/models
```

这不表示请求会发送给 OpenAI。模型仍在你自己的 GPU 上运行，只是接口形状兼容，所以许多现有客户端能直接接入。

## 6.2 启动服务

先在终端 A 运行：

```bash
conda activate vllm-env

vllm serve ~/llm-lab/models/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

参数含义：

- `--served-model-name`：客户端请求里填写的模型名；
- `--host 127.0.0.1`：只允许本机访问，初学和调试更安全；
- `--port 8000`：服务端口；
- `--max-model-len 4096`：限制上下文长度，降低 KV Cache 压力；
- `--gpu-memory-utilization 0.85`：给系统和其他开销留出空间。

看到服务器完成启动并监听端口后，不要关闭终端 A。

## 6.3 用 curl 检查服务

在终端 B 运行：

```bash
curl http://127.0.0.1:8000/v1/models
```

再发起对话：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [
      {"role": "system", "content": "你是一个简洁、严谨的助手。"},
      {"role": "user", "content": "什么是连续批处理？"}
    ],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Windows PowerShell 原生的 `curl` 在某些版本中可能是别名，命令转义也与 Bash 不同。本文 curl 示例应在 Linux/WSL/AutoDL 终端运行。

## 6.4 用 Python OpenAI SDK 调用

安装客户端：

```bash
pip install -U openai
```

创建 `~/llm-lab/scripts/call_vllm.py`：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-not-used",
)

response = client.chat.completions.create(
    model="qwen3-4b",
    messages=[
        {"role": "system", "content": "你是一个严谨的 AI 助手。"},
        {"role": "user", "content": "请解释 vLLM 和 SWIFT 的分工。"},
    ],
    temperature=0,
    max_tokens=256,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    },
)

print(response.choices[0].message.content)
```

运行：

```bash
python ~/llm-lab/scripts/call_vllm.py
```

`api_key` 在没有给 vLLM 服务配置鉴权时只是 SDK 要求的占位字符串。如果服务暴露到公网，不能继续使用无鉴权配置。

## 6.5 流式输出

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

stream = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "从零解释 KV Cache。"}],
    stream=True,
    max_tokens=512,
)

for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)
print()
```

流式输出改善的是用户看到文字的体验，不等于总生成时间一定缩短。

## 6.6 局域网或公网访问的安全问题

把 `--host` 改为 `0.0.0.0` 后，服务会监听所有网络接口。这样做之前至少考虑：

- 使用 `--api-key`（以当前 `vllm serve --help` 为准）或在反向代理层鉴权；
- 防火墙只开放需要的来源；
- 不要把无鉴权的 8000 端口直接暴露到公网；
- 设置请求大小、并发和超时限制；
- 日志中不要记录敏感 prompt。

---

# 7. vLLM 参数、显存与多卡

## 7.1 最常调整的参数

| 参数 | 作用 | 初学建议 |
|---|---|---|
| `model` | 模型 ID 或路径 | 先用本地明确路径 |
| `dtype` | 数值精度 | 新卡常用 `bfloat16`；不支持时尝试 `float16` |
| `max_model_len` | 最大上下文长度 | 先设 2048/4096，不要盲目开到模型上限 |
| `gpu_memory_utilization` | vLLM 使用显存的目标比例 | 从 0.80～0.90 开始 |
| `tensor_parallel_size` | 张量并行 GPU 数 | 必须与实际可见 GPU 数和模型结构兼容 |
| `max_num_seqs` | 同时调度的最大序列数 | OOM 时可降低 |
| `trust_remote_code` | 是否运行模型仓库自定义代码 | 只对可信仓库开启 |

## 7.2 生成参数怎么选

| 任务 | 推荐起点 |
|---|---|
| 事实问答、抽取、代码修复 | `temperature=0` |
| 普通对话 | `temperature=0.6, top_p=0.9` |
| 创意写作、多样采样 | `temperature=0.8, top_p=0.95` |
| 批量评测 | 固定参数，通常 `temperature=0` |

温度越高，输出通常越随机；`top_p` 限制参与采样的概率质量范围。不要一开始同时乱调五六个参数，否则不知道变化来自哪里。

## 7.3 为什么上下文越长越吃显存

推理显存主要由以下部分组成：

```text
模型权重 + KV Cache + CUDA/Kernel 工作区 + 临时张量
```

KV Cache 会随以下因素增长：

- 输入加输出的总 token 数；
- 同时活跃的请求数；
- 模型层数、KV 头数和 head dimension；
- KV Cache 数据类型。

因此 OOM 时最有效的第一批操作通常是：

1. 降低 `max_model_len`；
2. 降低 `gpu_memory_utilization`，为非 vLLM 开销留空间；
3. 降低并发或 `max_num_seqs`；
4. 关闭占 GPU 的其他进程；
5. 换小模型或使用受支持的量化模型。

## 7.4 多卡张量并行

两张 GPU 部署一个放不进单卡的模型：

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve ~/llm-lab/models/Qwen3-4B \
  --served-model-name qwen3-4b \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

这里是**张量并行**：一个模型的层内矩阵计算被分到两张卡。它不同于训练中常见的数据并行：

- 张量并行：多卡共同服务一个模型副本；
- 数据并行：每张卡有一个模型副本，分别处理不同数据。

卡数增加不代表速度线性增加。GPU 间通信、PCIe/NVLink 带宽、请求规模都会影响收益。

## 7.5 量化模型

量化把 BF16/FP16 权重压到 8 bit、4 bit 或其他格式，可以减少权重显存，但要注意：

- 量化格式必须受当前 vLLM 和 GPU 支持；
- “4bit”不是一个统一格式，AWQ、GPTQ、bitsandbytes 等不能混为一谈；
- 量化可能影响精度和速度；
- 不要只改一个 `--quantization` 参数，就期待普通 BF16 权重自动变成正确量化权重。

初学时先用小型 BF16/FP16 模型跑通全流程，再单独学习量化。

---

# 8. SWIFT 入门：安装和认识命令

## 8.1 安装 ms-swift

```bash
conda activate swift-env
python -m pip install -U pip
pip install -U "ms-swift[llm]"
```

检查：

```bash
swift --help
swift sft --help
python -c "import swift; print(swift.__version__)"
```

`swift` 这个可执行命令必须来自当前 `swift-env`。可以运行：

```bash
which swift
which python
```

如果需要 DeepSpeed 多卡训练，再安装与当前 PyTorch/CUDA 兼容的版本：

```bash
pip install -U deepspeed
```

不要为了“版本越新越好”反复无条件升级 PyTorch、vLLM、FlashAttention。先保存当前可工作版本，再按 SWIFT 当前安装文档处理兼容矩阵。

## 8.2 常见子命令

| 命令 | 用途 |
|---|---|
| `swift sft` | 监督微调，本文主线 |
| `swift infer` | 交互或批量推理，适合检查模型/LoRA |
| `swift deploy` | 启动推理服务，可选择推理后端 |
| `swift export` | 合并 LoRA、导出模型等 |
| `swift dpo` / `swift rlhf` | 偏好优化或强化学习入口，具体名称看版本 |
| `swift pt` | 继续预训练 |

最重要的排错命令是：

```bash
swift sft --help | less
swift infer --help | less
swift export --help | less
swift deploy --help | less
```

## 8.3 先用内置数据做最小烟雾测试

在用自己的数据前，可以用当前 SWIFT 文档提供的小型内置/Hub 数据集跑几十到几百条。示意：

```bash
swift sft \
  --model ~/llm-lab/models/Qwen3-4B \
  --train_type lora \
  --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#200' \
  --num_train_epochs 1 \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --logging_steps 5 \
  --save_steps 20 \
  --output_dir ~/llm-lab/output/smoke-test
```

数据集 ID 及切片语法会随版本和数据源支持情况变化。如果这个示例数据不可用，直接进入下一节，用本地 JSONL 最可靠。

---

# 9. 准备自己的 SFT 数据集

## 9.1 推荐使用 messages 格式 JSONL

JSONL 表示“一行一个 JSON 对象”。文件 `~/llm-lab/data/train.jsonl` 可以是：

```jsonl
{"messages":[{"role":"system","content":"你是实验室安全规范助手。"},{"role":"user","content":"进入激光实验区前要做什么？"},{"role":"assistant","content":"先完成安全培训，佩戴规定的激光防护眼镜，确认光路遮挡与警示标识，并按实验室流程登记。"}]}
{"messages":[{"role":"system","content":"你是实验室安全规范助手。"},{"role":"user","content":"发现设备异常发热怎么办？"},{"role":"assistant","content":"立即停止实验并按规程断电，保持安全距离，通知设备负责人；不要在未排查原因前重新上电。"}]}
{"messages":[{"role":"user","content":"用一句话解释监督微调。"},{"role":"assistant","content":"监督微调是用输入与标准回答组成的数据继续训练模型，使模型学会特定任务、知识表达或回答风格。"}]}
```

注意：真正的 JSONL 文件中，每个样本必须独占一行，不能在一个 JSON 对象内部随意换行。

## 9.2 role 的含义

- `system`：定义整体身份、规则或风格，可以没有；
- `user`：用户输入；
- `assistant`：目标回答，训练损失通常主要计算在这一部分；
- 多轮对话可以继续追加 `user`、`assistant`。

多轮示例：

```json
{"messages":[
  {"role":"system","content":"你是 Python 教师。"},
  {"role":"user","content":"列表和元组有什么区别？"},
  {"role":"assistant","content":"列表可变，元组通常不可变。"},
  {"role":"user","content":"各给一个字面量例子。"},
  {"role":"assistant","content":"列表是 [1, 2]，元组是 (1, 2)。"}
]}
```

写进 JSONL 时仍要压成单行。

## 9.3 划分训练集和验证集

不要用完全相同的数据同时训练和验证。一个简单起点是：

- 80%～95% 做训练；
- 5%～20% 做验证；
- 另留一组从未参与调参的问题做最终测试。

文件：

```text
~/llm-lab/data/train.jsonl
~/llm-lab/data/val.jsonl
~/llm-lab/data/test_questions.jsonl
```

数据很少时，验证指标波动会很大，不能仅凭一条回答判断训练成功。

## 9.4 检查 JSONL 语法

创建 `~/llm-lab/scripts/check_jsonl.py`：

```python
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
count = 0

with path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"第 {line_number} 行缺少非空 messages")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"第 {line_number} 行 role 非法: {message}")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"第 {line_number} 行 content 不是字符串")
        count += 1

print(f"检查通过，共 {count} 条样本: {path}")
```

运行：

```bash
python ~/llm-lab/scripts/check_jsonl.py ~/llm-lab/data/train.jsonl
python ~/llm-lab/scripts/check_jsonl.py ~/llm-lab/data/val.jsonl
```

## 9.5 数据质量比堆数量更重要

检查以下问题：

- 回答是否事实正确；
- 相似问题的答案是否互相矛盾；
- system 规则是否稳定；
- 是否包含密码、个人隐私或不应训练的数据；
- 答案格式是否正是你希望模型模仿的格式；
- 是否存在大量重复样本；
- 超长样本是否超过 `max_length` 被截断。

LoRA 不能可靠地把低质量数据变成高质量知识。

---

# 10. 用 SWIFT 做 LoRA 微调

## 10.1 第一次训练使用保守配置

```bash
conda activate swift-env

swift sft \
  --model ~/llm-lab/models/Qwen3-4B \
  --train_type lora \
  --dataset ~/llm-lab/data/train.jsonl \
  --val_dataset ~/llm-lab/data/val.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --logging_steps 5 \
  --eval_steps 20 \
  --save_steps 20 \
  --save_total_limit 2 \
  --output_dir ~/llm-lab/output/my-first-lora \
  --warmup_ratio 0.05
```

如果 GPU 不支持 BF16，把 `--torch_dtype bfloat16` 改为当前硬件和版本支持的 FP16 配置。具体参数名称以 `swift sft --help` 为准。

## 10.2 每个核心参数是什么意思

### 模型和数据

- `--model`：基础模型 ID 或本地目录；
- `--dataset`：训练数据；
- `--val_dataset`：验证数据；
- `--output_dir`：检查点、训练参数和日志的根目录。

### 训练方式

- `--train_type lora`：只训练 LoRA 参数；
- `--target_modules all-linear`：给主要线性层加 LoRA，覆盖广但训练参数也更多；
- `--lora_rank 8`：LoRA 的秩，越大容量越高、参数越多；
- `--lora_alpha 32`：LoRA 更新的缩放系数；
- `--lora_dropout 0.05`：LoRA 分支的 dropout。

### 批量大小

- `per_device_train_batch_size=1`：每张卡每个前向步骤放 1 条样本；
- `gradient_accumulation_steps=16`：累积 16 次梯度后再更新参数；
- 单卡近似有效批量大小：`1 × 16 = 16`；
- 多卡数据并行近似有效批量大小：`每卡批量 × 累积步数 × GPU 数`。

梯度累积能降低单步显存，但不会让总计算量消失。

### 长度和轮数

- `max_length=1024`：训练样本 Tokenizer 处理后的最大 token 长度；
- `num_train_epochs=1`：完整遍历训练集一次；
- 长度从 1024 增到 2048 往往显著增加显存和计算量。

### 学习率和保存

- `learning_rate=1e-4`：LoRA 常用量级的起点，不是所有任务的最佳值；
- `logging_steps`：每隔多少更新步骤打印训练日志；
- `eval_steps`：每隔多少步验证；
- `save_steps`：每隔多少步保存；
- `save_total_limit=2`：限制检查点数量，避免磁盘被占满；
- `warmup_ratio=0.05`：开始一小段时间逐渐增加学习率。

## 10.3 OOM 时按什么顺序改

第一次 OOM 不要随机删参数。按以下顺序：

1. 确认 GPU 没有其他进程：`nvidia-smi`；
2. 保持 `per_device_train_batch_size=1`；
3. 把 `max_length` 从 2048 降到 1024 或 512；
4. 启用 SWIFT 当前版本支持的 gradient checkpointing；
5. 换更小模型；
6. 使用 QLoRA/4bit，但要单独确认量化后端和 GPU 兼容性；
7. 最后再设计多卡方案。

不要把 `gradient_accumulation_steps` 当成直接控制单步显存的主要开关；它主要改变有效批量和参数更新频率。

## 10.4 怎么判断训练正在正常进行

观察：

- `loss` 是否是有限数值，不是 `nan`；
- 是否按设置输出日志、验证并保存检查点；
- GPU 利用率和显存是否正常；
- 每秒样本数或每步耗时是否稳定；
- 输出目录是否出现 `checkpoint-*`。

Loss 下降表示模型更能拟合训练目标，但不自动代表事实正确、泛化良好。必须用未参与训练的问题测试。

## 10.5 找到实际检查点

```bash
find ~/llm-lab/output/my-first-lora -maxdepth 3 -type f \
  -name adapter_model.safetensors -print
```

假设输出为：

```text
/home/user/llm-lab/output/my-first-lora/v0-.../checkpoint-100/adapter_model.safetensors
```

那么传给 `--adapters` 的是它所在的目录：

```text
/home/user/llm-lab/output/my-first-lora/v0-.../checkpoint-100
```

不是 `.safetensors` 文件本身。

## 10.6 多卡训练的基本概念

SWIFT 可结合 PyTorch 分布式或 DeepSpeed。多卡之前先弄清目标：

- 模型单卡放得下，只想提高吞吐：数据并行；
- 单卡放不下全部训练状态：ZeRO-2/ZeRO-3、FSDP 等分片方案；
- GRPO 还需要 rollout 引擎：训练卡和推理卡可能要分工。

多卡命令、DeepSpeed 配置和设备划分随 SWIFT 版本变化较快。先用单卡小数据跑通，再查看当前版本示例：

```bash
swift sft --help | grep -i -E "deepspeed|device|distributed"
```

不要在没理解进程数的情况下同时套 `torchrun`、DeepSpeed launcher 和 SWIFT 自己的启动方式，否则容易一张卡启动多个进程并立即 OOM。

---

# 11. 用 SWIFT 测试 LoRA 效果

## 11.1 先测试基础模型

```bash
conda activate swift-env

swift infer \
  --model ~/llm-lab/models/Qwen3-4B \
  --stream true \
  --temperature 0 \
  --max_new_tokens 256
```

在交互界面输入几条事先准备好的测试题并保存回答。

## 11.2 再测试 LoRA

```bash
swift infer \
  --model ~/llm-lab/models/Qwen3-4B \
  --adapters /实际路径/checkpoint-100 \
  --stream true \
  --temperature 0 \
  --max_new_tokens 256
```

有些 SWIFT 版本能从 adapter 配置自动找回基础模型，但明确写出 `--model` 更容易理解和排错。若帮助中说明二者不能同时传，以当前版本 `swift infer --help` 为准。

## 11.3 公平对比规则

微调前后必须保持：

- 相同 system prompt；
- 相同 user prompt；
- 相同 Chat Template；
- 相同温度、最大生成长度和停止条件；
- 对随机采样任务固定随机种子或重复多次。

建议做一张表：

| 问题 | 基础模型回答 | LoRA 回答 | 期望答案 | 判定 |
|---|---|---|---|---|
| 未见过的问题 1 | ... | ... | ... | 通过/失败 |
| 未见过的问题 2 | ... | ... | ... | 通过/失败 |

训练集原题答对只能证明记住了；未见过但同分布的问题答得更好，才是更有价值的信号。

## 11.4 过拟合的常见表现

- 不管问什么都复读训练答案；
- 强行套用某个格式；
- 训练 loss 很低，验证 loss 上升；
- 训练集效果很好，测试集变差；
- 一小份数据训练过多 epoch 后，通用能力明显下降。

可以尝试减少 epoch、降低学习率、增加高质量多样数据、减小 LoRA rank，或更仔细设计验证集。

---

# 12. 合并 LoRA，并交给 vLLM 部署

## 12.1 什么叫合并 LoRA

训练时实际有效权重可以理解为：

```text
有效权重 = 基础权重 + LoRA 学到的增量
```

合并会把增量写回完整模型权重，产出一个可以像普通模型一样加载的新目录。

优点：部署简单、兼容性通常更好。缺点：需要额外磁盘空间，之后切换不同 LoRA 不够灵活。

## 12.2 用 SWIFT 合并

先查看当前版本：

```bash
swift export --help | grep -i -E "adapter|merge|output"
```

常见命令形式如下：

```bash
conda activate swift-env

swift export \
  --model ~/llm-lab/models/Qwen3-4B \
  --adapters /实际路径/checkpoint-100 \
  --merge_lora true \
  --output_dir ~/llm-lab/models/Qwen3-4B-my-lora-merged
```

某些版本会自动生成带 `-merged` 的输出目录，或把参数命名为 `--output_dir`/`--output_path`。以 `swift export --help` 的实际结果为准，不要凭空猜输出位置。

## 12.3 检查合并产物

```bash
ls -lh ~/llm-lab/models/Qwen3-4B-my-lora-merged
du -sh ~/llm-lab/models/Qwen3-4B-my-lora-merged
```

它应包含完整权重、配置和 Tokenizer 文件，大小接近基础模型，而不是只有几十 MB。

## 12.4 用 vLLM 部署合并模型

切换到 vLLM 环境：

```bash
conda activate vllm-env

vllm serve ~/llm-lab/models/Qwen3-4B-my-lora-merged \
  --served-model-name my-lab-assistant \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

测试：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-lab-assistant",
    "messages": [{"role":"user","content":"进入激光实验区前要做什么？"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

这一步把 SWIFT 训练和 vLLM 部署完整接起来了。

---

# 13. 不合并 LoRA 的部署方式

## 13.1 使用 vLLM 的 LoRA 支持

vLLM 可以在基础模型上动态挂载 LoRA。常见服务命令：

```bash
conda activate vllm-env

vllm serve ~/llm-lab/models/Qwen3-4B \
  --served-model-name qwen3-base \
  --enable-lora \
  --lora-modules my-lora=/实际路径/checkpoint-100 \
  --max-lora-rank 8 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096
```

客户端请求中的 `model` 通常使用 LoRA 模块名 `my-lora`。具体模型列表先查：

```bash
curl http://127.0.0.1:8000/v1/models
```

不同 vLLM 版本对动态 LoRA、最大 rank、模块路径和请求模型名的行为可能变化，部署前应查看：

```bash
vllm serve --help | grep -i lora
```

## 13.2 使用 SWIFT Deploy

当 `swift-env` 已安装与 SWIFT 兼容的 vLLM 后端时，可用类似：

```bash
conda activate swift-env

swift deploy \
  --model ~/llm-lab/models/Qwen3-4B \
  --adapters /实际路径/checkpoint-100 \
  --infer_backend vllm \
  --served_model_name my-lora \
  --host 127.0.0.1 \
  --port 8000
```

参数拼写以 `swift deploy --help` 为准。这个方式的好处是训练和部署都在 SWIFT 的参数体系中；直接 `vllm serve` 则更接近标准 vLLM 运维方式。

## 13.3 合并还是不合并

| 需求 | 建议 |
|---|---|
| 只部署一个 LoRA，追求简单 | 合并后部署 |
| 一个基础模型切换多个 LoRA | 动态加载 LoRA |
| 目标引擎不支持该 LoRA 配置 | 合并后部署 |
| 磁盘空间紧张 | 尽量不合并 |
| 需要把完整模型交付给另一套系统 | 合并，并做完整验证 |

---

# 14. SWIFT 的 GRPO 强化学习入口

这一节只建立地图，不建议在还没跑通 SFT 时直接开始。

## 14.1 GRPO 在做什么

对同一个问题，模型采样多个回答：

```text
问题
 ├─ 回答 A → 奖励 1.0
 ├─ 回答 B → 奖励 0.0
 ├─ 回答 C → 奖励 0.5
 └─ 回答 D → 奖励 1.0
```

GRPO 根据组内相对奖励构造优势，增加高奖励回答对应 token 的概率，降低低奖励回答的概率，同时使用 KL 等约束避免策略变化过猛。

## 14.2 为什么 GRPO 会用到 vLLM

每个 prompt 要生成多个完整回答，这一步叫 rollout。生成量通常很大，因此可用 vLLM 提高 rollout 吞吐。

典型资源包括：

- 正在训练的策略模型；
- 优化器、梯度和激活；
- 参考策略或 KL 所需信息；
- rollout 推理引擎和 KV Cache；
- 奖励函数或奖励模型。

所以 GRPO 通常比 LoRA SFT 更复杂、更吃资源。

## 14.3 开始 GRPO 前必须先回答的问题

1. 奖励怎么计算？数学题可以核验答案，开放式问答不能随便用字符串相等。
2. 数据只有 prompt，还是包含标准答案/校验字段？
3. 每个 prompt 采样几个回答？生成多长？
4. rollout 和训练是否共用 GPU？
5. 当前 SWIFT 版本使用 `swift rlhf --rlhf_type grpo`，还是提供独立入口？
6. vLLM、PyTorch 和 SWIFT 版本是否兼容？

先运行：

```bash
swift --help
swift rlhf --help
```

再基于当前版本的官方示例启动。不要照抄旧版 GRPO 命令，因为 reward 参数名、vLLM 模式、设备划分和入口变化较快。

## 14.4 推荐学习顺序

```text
SFT + LoRA
  ↓
理解采样参数和批量生成
  ↓
写一个可独立测试的 reward 函数
  ↓
用极小数据做 GRPO smoke test
  ↓
检查 reward、KL、completion length 和 loss
  ↓
再扩大数据、生成数量和模型规模
```

本目录已有 [SWIFT与GRPO强化学习_从零详解.md](./SWIFT与GRPO强化学习_从零详解.md)，可在完成本文 SFT 主线后继续阅读。

---

# 15. 一条完整工作流

下面是一份真正执行实验时可以照着走的清单。

## 阶段 A：环境

```bash
nvidia-smi
conda create -n vllm-env python=3.11 -y
conda create -n swift-env python=3.11 -y
```

vLLM 环境：

```bash
conda activate vllm-env
pip install -U pip
pip install vllm openai
```

SWIFT 环境：

```bash
conda activate swift-env
pip install -U pip
pip install -U "ms-swift[llm]" modelscope
```

## 阶段 B：模型

```bash
mkdir -p ~/llm-lab/{models,data,output,scripts}
modelscope download --model Qwen/Qwen3-4B \
  --local_dir ~/llm-lab/models/Qwen3-4B
```

## 阶段 C：原模型基线

```bash
conda activate vllm-env
vllm serve ~/llm-lab/models/Qwen3-4B \
  --served-model-name base \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096
```

用固定测试题保存原始回答，然后停止服务释放显存。

## 阶段 D：数据检查

```bash
conda activate swift-env
python ~/llm-lab/scripts/check_jsonl.py ~/llm-lab/data/train.jsonl
python ~/llm-lab/scripts/check_jsonl.py ~/llm-lab/data/val.jsonl
```

## 阶段 E：LoRA SFT

```bash
swift sft \
  --model ~/llm-lab/models/Qwen3-4B \
  --train_type lora \
  --dataset ~/llm-lab/data/train.jsonl \
  --val_dataset ~/llm-lab/data/val.jsonl \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --num_train_epochs 1 \
  --output_dir ~/llm-lab/output/my-first-lora
```

## 阶段 F：效果验证

```bash
swift infer \
  --model ~/llm-lab/models/Qwen3-4B \
  --adapters /实际路径/checkpoint-N \
  --temperature 0
```

使用与基线完全相同的未见测试题，对比结果。

## 阶段 G：合并和部署

```bash
swift export \
  --model ~/llm-lab/models/Qwen3-4B \
  --adapters /实际路径/checkpoint-N \
  --merge_lora true \
  --output_dir ~/llm-lab/models/my-merged-model
```

```bash
conda activate vllm-env
vllm serve ~/llm-lab/models/my-merged-model \
  --served-model-name my-model \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096
```

## 阶段 H：保留实验记录

至少保存：

```text
使用的基础模型和来源
pip freeze 或关键包版本
GPU 型号和数量
训练命令
数据版本/哈希
训练日志
选择哪个 checkpoint 以及原因
基线与微调后的测试结果
最终部署命令
```

---

# 16. 常见报错排查

## 16.1 `torch.cuda.is_available()` 是 `False`

检查：

```bash
nvidia-smi
which python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

常见原因：装了 CPU 版 PyTorch、激活错 Conda 环境、WSL GPU 未配置、驱动太旧或容器未映射 GPU。

## 16.2 `CUDA out of memory`

vLLM：

- 降低 `max_model_len`；
- 降低并发/`max_num_seqs`；
- 适当降低 `gpu_memory_utilization`；
- 关闭其他 GPU 进程；
- 换小模型或兼容量化模型。

SWIFT SFT：

- `per_device_train_batch_size=1`；
- 降低 `max_length`；
- 使用 gradient checkpointing；
- 换小模型；
- 再考虑 QLoRA、ZeRO/FSDP。

## 16.3 `Address already in use`

8000 端口已被占用：

```bash
ss -ltnp | grep 8000
```

可以停止原进程，或改成 `--port 8001`。客户端的 `base_url` 也要同步修改。

## 16.4 `model not found` 或请求返回模型名错误

区分两种名字：

- 服务启动时的模型路径：`~/llm-lab/models/Qwen3-4B`；
- 客户端请求使用的服务名：`--served-model-name qwen3-4b`。

先查询：

```bash
curl http://127.0.0.1:8000/v1/models
```

请求中的 `model` 必须使用服务实际列出的名字。

## 16.5 回答乱码、格式异常或一直续写

检查：

- Tokenizer 是否与模型匹配；
- 是否用了正确 Chat Template；
- 基础模型是否真的是 Instruct/Chat 版；
- 停止 token 是否配置正确；
- 是否错误地对已经套模板的文本再次套模板；
- LoRA 的基础模型是否与训练时一致。

## 16.6 Qwen 输出很长的思考内容

部分 Qwen 模型支持 thinking 模式。需要简洁回答时，通过模型支持的 Chat Template 参数关闭，例如：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

这属于模型模板行为，不是所有模型和版本都支持。不要把这个参数强行传给无此功能的模型。

## 16.7 SWIFT 不认识某个参数

不要继续复制更多旧命令。立即运行：

```bash
swift sft --help | grep -i 参数关键词
swift infer --help | grep -i 参数关键词
swift export --help | grep -i 参数关键词
```

再记录版本：

```bash
swift --version
pip show ms-swift
```

教程和本机版本不同是大模型工具链中很常见的问题。

## 16.8 安装 vLLM 后依赖冲突

先保存完整错误中的“哪个包要求哪个版本”，再检查：

```bash
pip check
pip list | grep -E "torch|vllm|transformers|swift"
```

最稳妥的处理通常是在新 Conda 环境中按兼容组合重装，而不是在旧环境里连续强制降级/升级。不要把 `pip check` 的冲突警告忽略后直接开始长时间训练。

## 16.9 模型下载失败

检查：

- 磁盘空间：`df -h`；
- 目录大小：`du -sh 模型目录`；
- 网络是否能访问对应 Hub；
- 私有/gated 模型是否需要登录或接受许可；
- 是否留下不完整缓存。

不要随意删除不确定的共享缓存。先确认路径属于自己的实验目录。

## 16.10 训练 loss 是 `nan`

可能原因：

- 学习率过高；
- FP16 溢出；
- 数据为空、全被截断或标签掩码异常；
- 输入含异常数值/格式；
- 不兼容的量化或自定义模型实现。

先用 5～20 条手工检查过的短样本、较低学习率和 BF16（硬件支持时）做最小复现。

## 16.11 vLLM 服务启动了但请求不通

按层排查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

再检查：

- 服务是否仍在运行；
- 客户端端口是否正确；
- 服务在容器/远程主机时是否做了端口映射；
- AutoDL/Jupyter 是否需要代理或 SSH 端口转发；
- 防火墙是否允许访问；
- `127.0.0.1` 指的是客户端自己还是服务器。

远程服务器可使用 SSH 转发：

```bash
ssh -L 8000:127.0.0.1:8000 用户名@服务器地址
```

然后本机客户端访问 `http://127.0.0.1:8000/v1`。

---

# 17. 练习路线与验收清单

## 练习 1：vLLM 离线推理

任务：一次输入 8 个不同问题，打印每个回答和生成 token 数。

验收：

- 8 个请求一次传给引擎；
- 输出顺序能与问题对应；
- 能解释 `max_tokens` 与 `max_model_len` 的区别。

## 练习 2：启动 API

任务：启动服务，用 curl 和 Python SDK 各调用一次。

验收：

- `/v1/models` 可访问；
- SDK 使用自己的 `base_url`；
- 能解释“OpenAI 兼容”不等于把数据发往 OpenAI。

## 练习 3：制作 50 条 SFT 数据

任务：围绕一个小而明确的主题制作数据，划分训练集/验证集。

验收：

- JSONL 每行都是合法 JSON；
- 使用 `messages`；
- 数据没有明显矛盾和重复；
- 有至少 10 条未参与训练的测试问题。

## 练习 4：LoRA SFT

任务：用小模型和 50 条数据完成 1～3 epoch 试验。

验收：

- 有 adapter 检查点；
- 能找到实际 checkpoint 路径；
- 记录了 loss 和训练命令；
- 未见问题上的回答至少在目标风格或任务上有可解释变化。

## 练习 5：合并与部署

任务：合并 LoRA，用 vLLM 提供服务。

验收：

- 合并目录包含完整权重和 Tokenizer；
- vLLM 能加载；
- API 回答与 SWIFT infer 基本一致；
- 知道何时应选择动态 LoRA，而不是合并。

## 最终自测问题

1. vLLM 主要优化延迟还是吞吐？为什么？
2. KV Cache 里缓存了什么？为什么上下文越长越占显存？
3. `max_tokens` 和 `max_model_len` 分别控制什么？
4. SWIFT、vLLM 和 Transformers 的分工是什么？
5. LoRA 检查点为什么不能脱离基础模型随意使用？
6. 有效训练 batch size 怎么估算？
7. 为什么训练 loss 下降不能证明实际任务效果一定变好？
8. 合并 LoRA 的优缺点是什么？
9. 为什么 GRPO 比普通 LoRA SFT 更吃资源？
10. 服务暴露到公网前必须补哪些安全措施？

如果不能用自己的话回答，回到对应章节再做一次小实验。

---

# 18. 常用命令速查

## 环境

```bash
nvidia-smi
conda activate vllm-env
conda activate swift-env
which python
pip check
```

## vLLM

```bash
python -c "import vllm; print(vllm.__version__)"

vllm serve 模型路径 \
  --served-model-name 模型服务名 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85

curl http://127.0.0.1:8000/v1/models
```

## SWIFT

```bash
swift --help
swift sft --help
swift infer --help
swift export --help
swift deploy --help

swift infer --model 模型路径
swift infer --model 模型路径 --adapters LoRA检查点目录
```

## 文件和资源

```bash
df -h
du -sh ~/llm-lab/models/*
find ~/llm-lab/output -name adapter_model.safetensors -print
ss -ltnp | grep 8000
```

---

# 19. 术语表

| 术语 | 零基础解释 |
|---|---|
| LLM | 大语言模型，根据已有 token 预测后续 token |
| Token | 模型处理文本的基本单位，不一定等于一个字或一个词 |
| Tokenizer | 在文本和 token ID 之间转换的组件 |
| Causal LM | 按从左到右方式预测后续 token 的生成模型 |
| Inference | 不更新参数，只让模型生成结果 |
| Training | 根据损失或奖励更新模型参数 |
| SFT | 用问题和标准答案做监督微调 |
| LoRA | 冻结基础权重，只训练低秩增量参数 |
| QLoRA | 在量化基础模型上训练 LoRA，以进一步降低显存 |
| Adapter | 附加到基础模型上的小型可训练模块/权重 |
| Checkpoint | 训练过程中保存的某个时刻的状态 |
| Merge | 把 LoRA 增量合入基础权重，生成完整模型 |
| Prompt | 提供给模型的输入文本或消息 |
| Chat Template | 把 system/user/assistant 消息转换成模型规定格式的模板 |
| Context Length | 输入和输出总共允许的 token 长度范围 |
| Prefill | 一次处理输入 prompt、建立初始 KV Cache 的阶段 |
| Decode | 每一步生成一个新 token 的阶段 |
| KV Cache | 缓存历史 token 的 Key/Value，避免重复计算 |
| Batch | 一次共同处理的多条数据或请求 |
| Throughput | 单位时间处理的总 token 数或请求数 |
| Latency | 单个请求等待结果所需时间 |
| TTFT | Time To First Token，首 token 延迟 |
| TPOT | Time Per Output Token，输出 token 间平均时间 |
| PagedAttention | vLLM 对 KV Cache 进行分页管理的关键机制 |
| Continuous Batching | 请求动态进入和退出批次的连续批处理 |
| Sampling | 按模型概率分布选择下一个 token |
| Temperature | 控制采样随机程度的参数 |
| top-p | 只从累计概率达到阈值的候选集合采样 |
| OOM | Out Of Memory，显存或内存不足 |
| BF16 / FP16 | 常见 16 位浮点格式 |
| Quantization | 用更低位宽表示权重或激活，减少资源使用 |
| Tensor Parallel | 把同一个模型的张量计算拆到多张 GPU |
| Data Parallel | 多个模型副本处理不同数据并同步梯度 |
| Gradient Accumulation | 多次前后向累积梯度后再更新参数 |
| Gradient Checkpointing | 用额外重算换取更少激活显存 |
| Epoch | 完整遍历一次训练集 |
| Learning Rate | 每次参数更新的步长 |
| Loss | 训练目标的误差量，用于反向传播 |
| Validation Set | 不参与参数更新，用于观察泛化的数据 |
| OpenAI-compatible API | 接口路径和请求格式兼容常见 OpenAI SDK 的本地/第三方服务 |
| Rollout | 当前策略模型针对 prompt 采样生成回答的过程 |
| Reward | 对生成结果质量给出的分数 |
| GRPO | 用组内多个回答的相对奖励优化策略的一类算法 |

---

## 最后记住四句话

1. **先用小模型、小数据、短上下文跑通，再扩大规模。**
2. **SWIFT 用来训练/微调，vLLM 用来高吞吐推理和部署。**
3. **所有命令都记录版本；参数不认识时先看本机 `--help`。**
4. **训练成功不等于效果成功，必须用未参与训练的测试集验证。**

