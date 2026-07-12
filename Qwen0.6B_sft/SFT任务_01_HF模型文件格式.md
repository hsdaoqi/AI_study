# HuggingFace 模型文件格式详解 · 以 Qwen3-0.6B 为例

> 任务背景:师兄任务 = AutoDL 多卡(Accelerate)+ Qwen3-0.6B SFT + wandb 上传曲线。
> 这是任务第①项:**看懂一个模型的各个文件**。配套脚本 `代码/inspect_qwen.py`(亲手解剖每个文件)。
> 参考:知乎《Hugging Face 和 Megatron 模型格式详解 - 以 Qwen3-0.6B 为例》

---

## 目录
- [0. 一句话总览](#0-一句话总览)
- [1. 模型目录里有哪些文件](#1-模型目录里有哪些文件)
- [2. config.json — 模型的身份证/形状](#2-configjson--模型的身份证形状)
- [3. model.safetensors — 权重(大脑)](#3-modelsafetensors--权重大脑)
- [4. tokenizer 相关文件 — 文字↔token](#4-tokenizer-相关文件--文字token)
- [5. generation_config.json — 生成默认参数](#5-generation_configjson--生成默认参数)
- [6. 加载与保存(代码)](#6-加载与保存代码)
- [7. Megatron 格式:为什么现在跳过](#7-megatron-格式为什么现在跳过)
- [8. 必记速查表](#8-必记速查表)

---

## 0. 一句话总览

> **一个 HF 模型 = 「形状说明书(config)+ 权重(safetensors)+ 文字翻译器(tokenizer)」。**
> 加载 = 读 config 建空壳 → 读 safetensors 填参数 → 读 tokenizer 配好分词。就两行代码。

```
config.json          → 告诉代码"模型长什么形状",用它建空壳
model.safetensors    → 往空壳里填参数(大脑)
tokenizer.*          → 文字 ↔ token id 的翻译器
generation_config    → 生成时的默认超参
```

对照你学过的 VAR:`config.json` ≈ VAR 的 `build_vae_var(depth=16, embed_dim=1024...)` 入参;
tokenizer ≈ VQVAE 那个"把图变 token"的翻译器,只不过这里把**文字**变 token。

---

## 1. 模型目录里有哪些文件

```
Qwen3-0.6B/
├── config.json               # 模型架构配置(必需)
├── model.safetensors         # 模型权重(必需)
├── tokenizer.json            # 完整 tokenizer 定义(必需)
├── tokenizer_config.json     # tokenizer 额外配置(必需)
├── vocab.json                # 词表:token字符串→id(BPE 必需)
├── merges.txt                # BPE 合并规则(BPE 必需)
├── generation_config.json    # 生成默认参数(推荐)
└── README.md / LICENSE       # 说明/许可(可选)
```

**最小可加载集**:config + safetensors + tokenizer.json + tokenizer_config.json + vocab.json + merges.txt。

---

## 2. config.json — 模型的身份证/形状

**作用**:定义模型架构参数,`from_pretrained` 靠它建空壳。**没有它模型加载不出来。**

Qwen3-0.6B 的关键字段:

| 字段 | 值 | 含义 |
|---|---|---|
| `architectures` | `["Qwen3ForCausalLM"]` | 用哪个模型类 |
| `num_hidden_layers` | **28** | Transformer 层数 |
| `hidden_size` | **1024** | 每个 token 向量的维度 |
| `num_attention_heads` | **16** | Q(查询)头数 |
| `num_key_value_heads` | **8** | KV 头数 → **GQA**,16/8=2,每 2 个 Q 头共享 1 组 KV |
| `intermediate_size` | 3072 | MLP 中间层宽度(≈ hidden 的 3 倍) |
| `vocab_size` | 151936 | 词表大小 |
| `torch_dtype` | bfloat16 | 权重精度 |
| `tie_word_embeddings` | true | 共享输入 embedding 和输出 lm_head |

> **GQA(Grouped Query Attention)是重点**:普通注意力 Q/K/V 头数相等;GQA 让多个 Q 头**共享**同一组 K/V,减少 KV cache 显存。所以你在权重里会看到 `q_proj` 大、`k_proj/v_proj` 小一半。

---

## 3. model.safetensors — 权重(大脑)

**作用**:存所有训练好的参数(几亿个浮点数)。**最大的文件。**

Qwen3-0.6B:**1.40 GB、751,632,384 参数(0.75B)、311 个张量、bf16**。

**safetensors vs 老的 .bin:**
| 格式 | 特点 |
|---|---|
| `.safetensors` ✅ | 新标准,安全(不能藏代码)、零拷贝加载快、支持 mmap |
| `.bin`(pytorch_model) | 老的 pickle 格式,能执行代码有安全风险 |

**参数命名规范**(HF 标准,看名字就知道是哪层哪块):
```
model.layers.{i}.self_attn.q_proj.weight
  └ model.layers.0 : 第0层
  └ self_attn      : 自注意力模块
  └ q_proj         : Query 投影
  └ weight         : 权重
```

**关键张量形状**(第 0 层):
```
model.embed_tokens.weight              [151936, 1024]   词嵌入表
model.layers.0.self_attn.q_proj.weight [2048, 1024]     Q: 16头×128
model.layers.0.self_attn.k_proj.weight [1024, 1024]     K: 8头×128 (GQA,少一半!)
model.layers.0.self_attn.v_proj.weight [1024, 1024]     V: 8头×128
model.layers.0.mlp.gate_proj.weight    [3072, 1024]     SwiGLU 门
model.norm.weight                      [1024]           最终归一化
lm_head.weight                         [151936, 1024]   输出到词表(与embed共享)
```

> 大模型(>5GB)会切成 `model-00001-of-000xx.safetensors` + 一个 `index.json` 记录哪个参数在哪片。0.6B 单文件放得下,不用管。

---

## 4. tokenizer 相关文件 — 文字↔token

把人话切成 token id 的规则。**等于 VQVAE 里"图→token",这里是"文字→token"。**

- **tokenizer.json**:核心,完整词表 + BPE 合并规则(几 MB)。
- **tokenizer_config.json**:配置 —— tokenizer 类、特殊 token、**chat template**。
- **vocab.json**:`token字符串 → id` 映射。
- **merges.txt**:BPE 合并规则,每行一条。

**特殊 token(SFT 关键)**:
```
bos_token = <|endoftext|>   句首
eos_token = <|im_end|>      句尾 ← SFT 里模型要学会答完生成它来"停下"
pad_token = <|endoftext|>   填充对齐用
```

**chat template(下一项 SFT 数据格式的引子)**:把一段多轮对话,按 Qwen 格式拼成一个字符串(带 `<|im_start|>role ... <|im_end|>`)。SFT 的训练数据就是这么拼出来的。

代码:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
ids = tok("你好,世界")["input_ids"]     # 文字 → id
tok.decode(ids)                          # id → 文字
```

---

## 5. generation_config.json — 生成默认参数

推理时 `model.generate()` 的默认设置:
```json
{ "temperature": 0.6, "top_k": 20, "top_p": 0.95, "do_sample": true,
  "eos_token_id": [151645, 151643] }
```
> 和你跑 VAR 推理传的 `top_k / top_p / cfg` 是同类东西——控制采样的随机性。

---

## 6. 加载与保存(代码)

**加载(SFT 脚本第一步):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="auto")
tok   = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```
`from_pretrained` 干三件事:**读 config 建空壳 + 读 safetensors 填权重 + 读 tokenizer 配置**。

**保存(SFT 训练完):**
```python
model.save_pretrained("output/my_sft_model")   # 存 config + safetensors
tok.save_pretrained("output/my_sft_model")      # 存 tokenizer 全套
```
→ 存出来的目录,和第 1 节那个结构一模一样,可以直接被别人 `from_pretrained` 加载。

---

## 7. Megatron 格式:为什么现在跳过

知乎文章后半大篇幅讲 **Megatron torch_dist 格式**(把所有层权重合并成 10 个大张量、支持 TP/PP 切分)。**你这个任务用不上,原因:**

- 回忆 DDP 笔记的区分:**DDP/Accelerate = 切数据**(每卡放完整模型);**Megatron = 切模型**(一个模型拆到多卡)。
- Qwen3-0.6B 一张卡放得下 → 只需切数据 → **Accelerate 的活,不是 Megatron 的活**。
- Megatron 那套(层合并、QKV 合并、TP/PP resharding)是**百亿级以上**大模型多机训练才用。

> 结论:知道"有 Megatron 这种切模型格式、大模型才用、能和 HF 互转"即可。**现在别学细节,是干扰。**

---

## 8. 必记速查表

| 文件 | 一句话作用 | Qwen3-0.6B 数字 |
|---|---|---|
| config.json | 模型形状 | 28层 / 1024维 / 16Q-8KV头(GQA) / 词表151936 |
| model.safetensors | 权重 | 1.40GB / 0.75B参数 / 311张量 / bf16 |
| tokenizer.json (+vocab/merges) | 文字↔token | BPE |
| tokenizer_config.json | 分词配置 | **eos=`<\|im_end\|>`** + chat template |
| generation_config.json | 生成默认参数 | temp0.6 top_k20 top_p0.95 |

**HF 命名规范**:`model.layers.{i}.{模块}.{子块}.weight`
**加载两行**:`AutoModelForCausalLM.from_pretrained` + `AutoTokenizer.from_pretrained`
**GQA 记忆**:16 个 Q 头共享 8 组 KV → k_proj/v_proj 比 q_proj 小一半 → 省 KV cache 显存
