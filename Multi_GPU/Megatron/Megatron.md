# 第一部分：3D 并行的底层数学与物理原理

Megatron-LM 最核心的贡献在于：**把 Transformer 模型中的矩阵乘法，物理上切开并分配到不同的 GPU 上计算**。

## 1. 张量并行（Tensor Parallelism, TP）
这是 Megatron-LM 的看家本领，主要针对 Transformer 的 **MLP 层** 和 **自注意力（Self-Attention）层**。

### ① MLP 层的切分（Column & Row Parallel）
标准的 MLP 层有两层全连接：第一层 $Y = \text{GeLU}(X W_1)$，第二层 $Z = Y W_2$。
Megatron-LM 将其设计为 **列并行（Column Parallel）** 紧跟 **行并行（Row Parallel）**。

```
【列并行：MLP Layer 1】
输入 X (所有 GPU 共享)
   │
   ├───> 传给 GPU 0 ───> 计算 Y_1 = GeLU(X · W_1_col0)
   │
   └───> 传给 GPU 1 ───> 计算 Y_2 = GeLU(X · W_1_col1)
```
* **实现**：将权重 $W_1$ 按列切开，分给 GPU 0 和 GPU 1。两张卡各自计算，**中途不需要任何通信**。

```
【行并行：MLP Layer 2】
GPU 0 的输入 Y_1 ───> 计算 Z_1 = Y_1 · W_2_row0 ──┐
                                                  ├───> [All-Reduce SUM] ───> 最终输出 Z
GPU 1 的输入 Y_2 ───> 计算 Z_2 = Y_2 · W_2_row1 ──┘
```
* **实现**：将权重 $W_2$ 按行切开。由于 $W_2$ 的输入是前面两张卡拼起来的 $[Y_1, Y_2]$，根据矩阵乘法原理，最终结果 $Z = Z_1 + Z_2$。
* **通信点**：两张卡算完各自的 $Z_1$ 和 $Z_2$ 后，进行一次 **All-Reduce (SUM)** 操作，在所有卡上同步求和，得到最终的 $Z$。

> 💡 **核心优势**：整个 MLP 块在计算过程中，**仅在最后进行了一次 All-Reduce 通信**，极大降低了网络开销。

---

### ② 自注意力（Self-Attention）层的切分
针对 Self-Attention，Megatron-LM 将 Query ($Q$)、Key ($K$)、Value ($V$) 投影矩阵按**注意力头（Attention Heads）**进行列并行切分：

```
GPU 0 负责计算前一半的 Heads (Q_0, K_0, V_0) ───> 算得 Attention_0
GPU 1 负责计算后一半的 Heads (Q_1, K_1, V_1) ───> 算得 Attention_1
```
* **输出投影层（Output Projection）**：接着使用**行并行**，将各自算好的 Attention 结果和各自对应的输出投影矩阵相乘，最后做一次 **All-Reduce** 累加。

---

## 2. 流水线并行（Pipeline Parallelism, PP）
当模型层数（Layers）极深时（例如 80 层），即使切分了矩阵，单卡也装不下。此时需要进行层间切分：
* GPU 0 负责前 40 层，GPU 1 负责后 40 层。

### 1F1B (One Forward, One Backward) 调度机制
简单的流水线会导致严重的“气泡”（即后方的 GPU 在等前方 GPU 算完，处于闲置状态）。
Megatron-LM 采用了 **1F1B 调度算法**：

```
GPU 0 (前层): [F1] ───> [F2] ───> [F3] ───> [B1] ───> [F4] ───> [B2] ...
GPU 1 (后层):         [F1] ───> [F2] ───> [B1] ───> [F3] ───> [B2] ...
```
* 在稳定状态下，每个 GPU 交替运行一个 Micro-batch 的前向传播（F）和一个反向传播（B），从而将激活值（Activations）的显存占用控制在极低的常数级别。

---

# 第二部分：核心数学公式——Batch Size 的分布式拆解

在 Megatron-LM 的配置文件里，你会看到以下四个参数，它们的关系决定了训练的吞吐量和显存：

1. **`tensor-model-parallel-size` (TP)**
2. **`pipeline-model-parallel-size` (PP)**
3. **`micro-batch-size` (MBS)**：单张卡单次前向传播处理的样本数。
4. **`global-batch-size` (GBS)**：整个集群在一次参数更新（Step）中处理的总样本数。

它们满足以下严格的数学公式（假设数据并行度为 DP）：$$\text{DP} = \frac{\text{Total GPUs}}{\text{TP} \times \text{PP}}$$
$$\text{GBS} = \text{MBS} \times \text{DP} \times \text{Gradient Accumulation Steps}$$

> 💡 **通俗理解**：如果你的 GBS 设得很大，而 MBS 设得很小，Megatron 会自动帮你计算并启用**梯度累积（Gradient Accumulation）**，以确保数学上的 batch size 准确。

---

# 第三部分：用 PyTorch 手写一个张量并行（TP）层

为了让你彻底看懂 Megatron 是如何控制 CUDA 的，我们不依赖任何大框架，仅用原生的 `torch.distributed` 来模拟写一个 **列并行全连接层（ColumnParallelLinear）**：

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    列并行线性层。
    输入 X: 形状 (Batch, In_Features)，所有 GPU 进程输入相同。
    权重 W: 在每张卡上只保存一部分列。
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # 1. 获取当前集群信息
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size() # 假设我们用 2 张卡，world_size=2
        
        # 2. 计算当前卡应该分配多少列
        assert out_features % self.world_size == 0
        self.local_out_features = out_features // self.world_size
        
        # 3. 初始化本地权重（每张卡只有完整权重的 1/N）
        self.weight = nn.Parameter(torch.randn(self.local_out_features, in_features))
        self.bias = nn.Parameter(torch.randn(self.local_out_features))

    def forward(self, x):
        # 4. 在每张卡上独立进行矩阵乘法。由于权重是转置的，所以按列乘
        # 输出形状: (Batch, local_out_features)
        out_parallel = F.linear(x, self.weight, self.bias)
        
        # 5. 【核心】：在真正的 Megatron 中，由于后面接行并行，此处不需要任何通信！
        # 如果我们需要直接拿完整结果，可以进行一次 All-Gather：
        # dist.all_gather(...) 将各个卡的结果拼回 (Batch, out_features)
        return out_parallel
```
Megatron-LM 的底层代码就是由大量这类经过精心设计的自定义 PyTorch/CUDA 算子构成的。

---

# 第四部分：手把手实操——在双卡服务器上启动 Megatron-LM

由于 Megatron-LM 不是一个可以通过 `pip install` 安装的普通库，而是一个**需要直接在其源码目录运行的项目**。我们将演示如何克隆代码、预处理数据，并在 **2 张 GPU** 上启动一个 GPT 模型的训练。

## 1. 环境依赖准备
Megatron 依赖高效的 Fused 算子，建议使用 NVIDIA 官方的 PyTorch 容器（如 `nvcr.io/nvidia/pytorch:23.10-py3`），或者在本地环境中安装：
```bash
# 开启加速
source /etc/network_turbo
# 克隆官方仓库
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

---

## 2. 准备二进制训练数据
在大规模训练中，频繁读取 JSON 或 TXT 文本会导致磁盘 I/O 成为严重瓶颈。Megatron 会先将文本**离线 token 化并打包成二进制 `.bin` 和 `.idx` 索引文件**，实现内存映射高速读取。

我们准备一个玩具文本文件 `toy.txt`：
```text
Megatron-LM is a highly optimized library for large-scale language models.
It implements tensor and pipeline parallelism for efficient training.
We are practicing multi-GPU training on a 2-GPU node.
```

下载 GPT2 的分词器文件（或使用你本地的任意分词器）：
```bash
# 下载 gpt2 的 vocab 和 merges
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

运行数据预处理脚本：
```bash
python tools/preprocess_data.py \
       --input toy.txt \
       --output-prefix my_gpt_dataset \
       --vocab-file gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
```
运行后，你的当前目录下会生成两个高效的二进制数据文件：
* `my_gpt_dataset_text_document.bin`
* `my_gpt_dataset_text_document.idx`

---

## 3. 编写双卡（TP=2）GPT 预训练脚本

现在，我们在 `Megatron-LM` 目录下新建一个运行脚本 `run_gpt_2card.sh`：

```bash
#!/bin/bash
# 限制 CUDA 的最大连接数，防止多卡通信死锁
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 1. 硬件分布式参数配置 (单机双卡)
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

# 2. 3D 并行度配置
# 我们一共有 2 张 GPU，设置 TP=2，PP=1。
# 意味着模型的所有层都装在两张卡上，但每一层都被矩阵切开了。
TP_SIZE=2
PP_SIZE=1

# 3. 极小模型架构配置 (适合双卡快速跑通测试)
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
SEQ_LENGTH=1024

# 4. 数据和分词器路径（请确保路径正确）
DATA_PATH="./my_gpt_dataset_text_document"
VOCAB_FILE="./gpt2-vocab.json"
MERGE_FILE="./gpt2-merges.txt"
SAVE_CHECKPOINT_PATH="./checkpoints"

# 合并 PyTorch 分布式启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \ #单台机器启动的进程数
    --nnodes $NNODES \ #参与训练的服务器（节点）总数
    --node_rank $NODE_RANK \ #当前服务器的全局编号
    --master_addr $MASTER_ADDR \ #主节点的 IP 地址
    --master_port $MASTER_PORT #通信端口号
"

# 5. 启动主程序 pretrain_gpt.py
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \ #张量并行度（TP）
    --pipeline-model-parallel-size $PP_SIZE \ #流水线并行度（PP）
    --num-layers $NUM_LAYERS \ #模型的层数（深度）
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTN_HEADS \ #多头注意力的头数
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \ #位置编码的最大长度
    --micro-batch-size 4 \ #微批次大小（MBS）
    --global-batch-size 16 \ #全局批次大小（GBS）
    --train-iters 100 \ #总迭代步数
    --lr-decay-iters 100 \ #学习率衰减步数
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \ #数据读取实现方式为内存映射（Memory Map）
    --split 949,50,1 \ #数据集的划分比例
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \ #梯度裁剪阈值
    --bf16 \
    --save $SAVE_CHECKPOINT_PATH \
    --load $SAVE_CHECKPOINT_PATH \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 100 \
    --eval-iters 10
```

---

## 4. 运行与日志解读

给脚本赋予执行权限并启动：
```bash
chmod +x run_gpt_2card.sh
./run_gpt_2card.sh
```

由于你开启了双卡张量并行（TP=2），当代码跑起来后，你可以重点观察以下几个维度的日志，这也是你在向师兄汇报时的**核心干货**：

### 关键日志 1：进程组初始化
```text
> initializing tensor model parallel with size 2
> initializing pipeline model parallel with size 1
> initializing model parallel with size 2
```
* **解读**：表明 Megatron 成功识别了双卡环境，并将 TP（Tensor Parallel）初始化为 2。

### 关键日志 2：参数统计
```text
 > number of parameters on tensor model parallel rank 0: 124439808
```
* **解读**：在 TP=2 模式下，单张卡上实际承载的参数量（以 124M 模型的 rank 0 为例）。这证明了模型参数成功被切分，降低了单卡的显存负担。

### 关键日志 3：迭代吞吐量
```text
 iteration       10/     100 | consumed samples:           160 | elapsed time per iteration (ms): 120.5 | learning rate: 1.500E-04 | lm loss: 10.453
```
* **解读**：
  * `iteration 10/ 100`：当前运行到第 10 个 Step（一共跑 100 个 Step）。
  * `elapsed time per iteration (ms): 120.5`：平均每个 Step 耗时 120.5 毫秒。**在调整不同的 TP/PP 参数时，这个耗时是衡量硬件效率（TFLOPS）最关键的指标**。

---

# 第五部分：Megatron-LM 与 PyTorch Lightning 的核心心智模型对比

为了让你在研究生涯中能够游刃有余地在这两个框架之间切换，请务必建立起它们之间清晰的边界：

| 维度 | PyTorch Lightning | Megatron-LM |
| :--- | :--- | :--- |
| **对代码的修改** | 模块化改写。你依然在写普通的 PyTorch 算子，由框架来托管循环。 | 你必须直接使用 Megatron 内部定制的 `ColumnParallelLinear` 和 `RowParallelLinear` 重构整个 Transformer 的每一个模块。 |
| **数据并行的粒度** | **样本级别**。卡 0 和卡 1 看到的是完全不同的图片/句子。 | **词元（Token）与层间级别**。如果开启 TP，卡 0 和卡 1 看到的可能是**同一个 Token 的不同特征通道**。 |
| **调试复杂度** | 简单。单卡跑通后，基本可以直接多卡无痛运行。 | 极难。由于涉及张量级别的切分和复杂的 NCCL 通信，一旦出现维度不对或通信死锁，报错堆栈极其底层，需要顺着矩阵乘法逻辑去排查。 |

---

