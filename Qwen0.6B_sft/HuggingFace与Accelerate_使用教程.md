# HuggingFace(transformers) + Accelerate 使用教程

> **先厘清一件事**:"HuggingFace" 不是单一框架,是**一整套库**;**Accelerate 本身就是 HuggingFace 出的**一员。
> 所以你要学的两块是:
> - **transformers** —— 加载模型、分词、生成、(可选)训练。**日常打交道最多的就是它**。
> - **Accelerate** —— 让训练跑多卡。
>
> 定位:偏"怎么用"的 **API 全表 + 可复制配方**。两个框架都讲全,最后讲**HF 模型 + Accelerate 多卡**的标准组合(就是你 `train_sft.py` 的骨架)。
> 想要 Accelerate 的"为什么/原理"(自动挡类比、prepare 黑箱),看 `Accelerate框架_从零详解.md`;这份偏实操。

---

## 目录
- [第一部分 · HuggingFace 生态](#第一部分--huggingface-生态)
  - [1. HF 是什么:五个库的关系](#1-hf-是什么五个库的关系)
  - [2. 三大件:Config / Model / Tokenizer + Auto 家族](#2-三大件config--model--tokenizer--auto-家族)
  - [3. Tokenizer 用法详解](#3-tokenizer-用法详解)
  - [4. Model 用法详解(forward / 算 loss / generate)](#4-model-用法详解forward--算-loss--generate)
  - [5. 下模型 / 下数据(国内加速)](#5-下模型--下数据国内加速)
  - [6. datasets 库速用](#6-datasets-库速用)
  - [7. Trainer:HF 自带训练器(对比 Accelerate)](#7-trainerhf-自带训练器对比-accelerate)
- [第二部分 · Accelerate 详解](#第二部分--accelerate-详解)
  - [8. 安装 & 启动(launch 参数全表)](#8-安装--启动launch-参数全表)
  - [9. Accelerator 对象:构造参数 + 属性 + 方法全表](#9-accelerator-对象构造参数--属性--方法全表)
  - [10. 实战配方(Cookbook · 10 个)](#10-实战配方cookbook--10-个)
  - [11. Accelerate 排错](#11-accelerate-排错)
- [第三部分 · 两个框架合起来](#第三部分--两个框架合起来)
  - [12. 标准组合:HF 模型 → Accelerate 多卡训练](#12-标准组合hf-模型--accelerate-多卡训练)
  - [13. 完整可复制模板](#13-完整可复制模板)
- [第四部分 · 速查小抄](#第四部分--速查小抄)

---

# 第一部分 · HuggingFace 生态

## 1. HF 是什么:五个库的关系

HuggingFace 是一套配合使用的库,各管一段:

| 库 | 管什么 | 你什么时候用 |
|---|---|---|
| **transformers** | 模型 + 分词器(加载/前向/生成/训练) | ⭐天天用:加载 Qwen、算 loss、生成 |
| **tokenizers** | 底层高速分词(transformers 内部调它) | 一般不直接碰,`AutoTokenizer` 帮你用 |
| **datasets** | 加载/处理数据集 | 下 alpaca_zh、map 转格式 |
| **huggingface_hub** | 从 Hub 下载/上传模型和数据 | `snapshot_download`、`hf download` |
| **accelerate** | 让训练跑多卡/混合精度 | 多卡训练(你的 SFT) |

> 一句话串起来:**用 `datasets` 拿数据 → 用 `transformers` 加载模型和分词器 → 用 `accelerate` 把训练铺到多卡上**。它们都出自 HuggingFace,彼此无缝。
>
> Hub(huggingface.co)= 模型和数据的"应用商店",`from_pretrained("Qwen/Qwen3-0.6B")` 就是从这儿拉。

安装:
```bash
pip install transformers datasets accelerate safetensors
```

---

## 2. 三大件:Config / Model / Tokenizer + Auto 家族

一个 HF 模型 = **三样东西**(你在 `SFT任务_01` 拆过文件,这里讲"用代码怎么拿"):

| 三大件 | 是什么 | 加载类 |
|---|---|---|
| **Config** | 模型形状(层数/宽度/词表大小) | `AutoConfig` |
| **Model** | 网络结构 + 权重 | `AutoModelForCausalLM` 等 |
| **Tokenizer** | 文字 ↔ token id 的翻译器 | `AutoTokenizer` |

### `Auto` 家族:自动认模型类型

`AutoXxx` 会**读 config 自动选对具体类**(Qwen→Qwen3ForCausalLM,你不用记具体类名)。按"任务头"选:

| Auto 类 | 用途 |
|---|---|
| `AutoModel` | 裸主干(只出 hidden states,不带任务头) |
| `AutoModelForCausalLM` | ⭐**因果语言模型**(GPT/Qwen,做生成/SFT) |
| `AutoModelForSeq2SeqLM` | 编码-解码(T5/BART,翻译摘要) |
| `AutoModelForSequenceClassification` | 文本分类(情感等) |
| `AutoModelForTokenClassification` | 序列标注(NER) |
| `AutoModelForQuestionAnswering` | 抽取式问答 |

### `from_pretrained` / `save_pretrained`(万能的两个方法)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载(从 Hub 或本地路径都行)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,       # 新版用 dtype(旧版 torch_dtype 已废弃,见下方提示)
    # device_map="auto",        # 自动把模型铺到可用 GPU(单卡推理常用;多卡训练交给 accelerate,别用它)
    # trust_remote_code=True,   # 模型自带自定义代码时需要(有些模型要)
)

# 保存(训练后)
model.save_pretrained("output/my-model")
tok.save_pretrained("output/my-model")
```

> ⚠️ **你亲眼见过的那个 warning**:`torch_dtype is deprecated! Use dtype instead!`
> 新版 transformers 把 `torch_dtype=` 改名成 `dtype=`。功能一样,换个名就没警告了。老代码用 `torch_dtype` 也还能跑(只是提醒)。

> `from_pretrained` 干的三件事(= `inspect_qwen.py` 演示的):读 config 建空壳 → 读 safetensors 填权重 → 读 tokenizer 配置。

---

## 3. Tokenizer 用法详解

### 3.1 最常用:`tokenizer(...)` 把文字变 token
```python
enc = tok("你好,世界", return_tensors="pt")   # return_tensors:pt=PyTorch张量,None=python列表
print(enc["input_ids"])        # token id
print(enc["attention_mask"])   # 1=真token,0=padding
```

常用参数:
```python
tok(
    ["句子1", "句子2"],          # 可传一个或一批
    return_tensors="pt",        # "pt" / None
    padding=True,               # 补齐到本 batch 最长(或 "max_length")
    truncation=True,            # 超长截断
    max_length=512,             # 截断/补齐长度上限
)
```

### 3.2 反过来:`decode` 把 token 变回文字
```python
ids = tok("你好")["input_ids"]
tok.decode(ids, skip_special_tokens=True)     # 跳过 <|im_end|> 这类特殊 token
tok.batch_decode(batch_ids, skip_special_tokens=True)   # 一批一起解
```

### 3.3 特殊 token
```python
tok.eos_token, tok.eos_token_id    # 结束符(Qwen 是 <|im_end|>)
tok.pad_token, tok.pad_token_id    # 补齐符
tok.bos_token                      # 起始符(有的模型有)
if tok.pad_token is None:          # 有的模型没设 pad,拿 eos 兜底
    tok.pad_token = tok.eos_token
```

### 3.4 ⭐`apply_chat_template`:把对话拼成模型认的格式(SFT/推理必用)
```python
messages = [
    {"role": "system", "content": "你是助手。"},
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
]

# 训练:要完整对话的 token
ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                              return_dict=False)      # 你踩过的坑:新版默认返回 Encoding,加 return_dict=False 拿列表

# 推理:只到"该助手答了"为止(add_generation_prompt=True),让模型接着往下生成
prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt")
```
- `tokenize=True` 直接出 token id;`tokenize=False` 出字符串。
- `add_generation_prompt=True` 会在末尾补 `<|im_start|>assistant\n`,推理时用。
- Qwen3 额外支持 `enable_thinking=False`(关思考模式,直接答)。
- **别自己手拼特殊 token**,一定用这个方法(它读 `tokenizer_config.json` 里的模板,和训练时一致)。

---

## 4. Model 用法详解(forward / 算 loss / generate)

### 4.1 前向 + 算 loss(训练用)
```python
out = model(input_ids=ids, attention_mask=mask, labels=labels)
loss = out.loss        # 传了 labels,HF 内部自动 shift + 交叉熵(见 SFT任务_02 Part B)
logits = out.logits    # [batch, seq_len, vocab] 每个位置对每个词的分数
```
- **传 `labels` 就自动算 loss**——这是 SFT 脚本只写一行 `out.loss` 的原因。
- `labels` 里填 `-100` 的位置不算 loss(SFT 的掩码就靠它)。

### 4.2 生成(推理用):`model.generate`
```python
model.eval()
with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=200,        # 最多生成多少新 token
        do_sample=False,           # False=贪心(可复现);True=采样(有随机性)
        temperature=0.7,           # 采样温度(越高越随机;do_sample=True 时生效)
        top_p=0.9,                 # 核采样
        top_k=50,                  # top-k 采样
        repetition_penalty=1.1,    # >1 抑制复读
        num_beams=1,               # >1 开 beam search
        pad_token_id=tok.pad_token_id or tok.eos_token_id,   # 防 padding 警告
    )
# 只取新生成的部分(去掉输入的 prompt)
new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
print(tok.decode(new_ids, skip_special_tokens=True))
```
> 对比效果就用这个(你的 `chat_compare.py` 里就是这套)。`do_sample=False` 贪心解码结果稳定,做前后对比最公平。

### 4.3 常用输出字段
| 字段 | 是什么 |
|---|---|
| `out.loss` | 传了 labels 才有;标量,可 `.backward()` |
| `out.logits` | 每个位置对词表的原始分数 |
| `out.hidden_states` | 各层隐藏态(需 `output_hidden_states=True`) |
| `out.past_key_values` | KV cache(生成时加速用) |

### 4.4 训练/推理模式 + 省显存
```python
model.train()                     # 训练模式(开 dropout 等)
model.eval()                      # 推理模式
model.config.use_cache = False    # 训练时关 KV cache(省显存,配合梯度检查点)
model.gradient_checkpointing_enable()   # 用时间换显存(大模型/长序列时开)
```

---

## 5. 下模型 / 下数据(国内加速)

AutoDL/国内直连 huggingface.co 慢,**用镜像**:

```bash
# 方式1:环境变量走镜像(最省心,对 from_pretrained / load_dataset 都生效)
export HF_ENDPOINT=https://hf-mirror.com

# 方式2:AutoDL 学术加速(能直连 HF)
source /etc/network_turbo

# 有 token 提高限速(可选;你日志里那个 "unauthenticated requests" 警告就是没设它)
export HF_TOKEN=hf_xxx
```

主动下载(不等训练时下):
```bash
# 命令行下整个模型
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
```
```python
# 或代码里下,返回本地路径
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen3-0.6B")
```
- 缓存默认在 `~/.cache/huggingface/`。想换位置:`export HF_HOME=/root/autodl-tmp/hf`。

---

## 6. datasets 库速用

```python
from datasets import load_dataset

ds = load_dataset("shibing624/alpaca-zh", split="train")   # 加载(走 HF_ENDPOINT 镜像)
print(ds)                    # 看有多少条、什么字段
print(ds[0])                 # 看第一条

ds = ds.select(range(10000)) # 只取前 1 万条
ds = ds.filter(lambda x: len(x["output"]) > 5)             # 过滤
ds = ds.map(lambda x: {"text": x["instruction"] + x["output"]})   # 加工/加字段
ds.to_json("out.jsonl", force_ascii=False)                 # 存成 jsonl
```
- 大数据集内存放不下:`load_dataset(..., streaming=True)` 流式读。
- **你的 `prepare_data.py` 就是**:`load_dataset` → `map`/循环转成 messages → 写 jsonl。

---

## 7. Trainer:HF 自带训练器(对比 Accelerate)

transformers 还自带一个**全自动训练器 `Trainer`**,几行就能训,连多卡/混合精度/日志都封装好:

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=20,
    report_to="wandb",          # 直接接 wandb
    save_strategy="epoch",
)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
trainer.train()
```

**Trainer vs 自己写 Accelerate 循环:**

| | Trainer | 自己写 Accelerate 循环(你现在这种) |
|---|---|---|
| 上手 | 最快,几行搞定 | 要自己写循环 |
| 透明度 | 黑盒(loss/循环都藏起来) | **每步都看得见,想改就改** |
| 灵活性 | 改行为要读源码/回调 | 完全自由 |
| 学习价值 | 低(不知道里面干嘛) | **高**(懂 forward/backward/step 全流程) |

> 💡 **师兄让你用 Accelerate 自己写循环,而不是 Trainer,就是要你搞懂训练细节**(loss 掩码、梯度累积、多卡)。Trainer 适合以后"只想快速出结果"时用。**底层其实 Trainer 也是用 Accelerate 实现的**——你现在学的是更底层、更通用的那层。

---

# 第二部分 · Accelerate 详解

> 原理(自动挡类比、prepare 黑箱、和 DDP 的关系)看 `Accelerate框架_从零详解.md`。这里是**API 全表 + 可复制配方**,写代码时对着抄。

## 8. 安装 & 启动(launch 参数全表)

### 8.1 安装
```bash
pip install accelerate
accelerate env        # 打印当前环境(torch版本/卡数/混合精度),排错时先跑这个
```

### 8.2 三种启动方式(对比记忆)

| 命令 | 起几个进程 | 什么时候用 |
|---|---|---|
| `python train.py` | 1 | 单卡 / 调试 |
| `torchrun --nproc_per_node=2 train.py` | 2 | 你 VAR/DDP 用的原生启动器 |
| `accelerate launch --num_processes 2 train.py` | 2 | **Accelerate 专用,推荐**(和 config 联动) |

> 三者底层机制一样:把脚本复制成 N 份同时跑,给每份设好"你是几号进程"的环境变量。

### 8.3 `accelerate launch` 参数全表

```bash
accelerate launch \
    --multi_gpu \                     # 声明多卡模式
    --num_processes 2 \               # 用几张卡(=几个进程)
    --num_machines 1 \                # 几台机器(单机=1)
    --mixed_precision bf16 \          # no / fp16 / bf16
    --gradient_accumulation_steps 4 \ # 梯度累积(也可在代码里设)
    --main_process_port 29500 \       # 主进程端口(被占用报错时换)
    --gpu_ids 0,1 \                   # 指定用哪几张卡(默认全用)
    train.py --your_arg xxx           # 你脚本自己的参数跟在最后
```

| 参数 | 作用 |
|---|---|
| `--num_processes N` | 进程数 = 卡数 |
| `--multi_gpu` | 显式多卡(更稳,避免被当单卡) |
| `--mixed_precision {no,fp16,bf16}` | 混合精度 |
| `--gradient_accumulation_steps N` | 梯度累积 |
| `--cpu` | 强制 CPU(本机没卡时体验多进程) |
| `--gpu_ids 0,1` | 只用指定卡 |
| `--main_process_port` | 换端口(`address already in use` 时) |
| `--num_machines`/`--machine_rank`/`--main_process_ip` | 多机训练才用 |
| `--config_file xxx.yaml` | 用指定配置文件 |

### 8.4 `accelerate config`(可选,配一次省事)
```bash
accelerate config          # 交互式问:几卡?bf16?多机?→ 存成默认配置
accelerate launch train.py # 之后直接 launch,不用带一堆参数
accelerate config default  # 想跳过交互、直接生成一个默认配置
```
- 配置存在:`~/.cache/huggingface/accelerate/default_config.yaml`
- **AutoDL 上更推荐直接命令行传参**(`--num_processes 2 --mixed_precision bf16`),省得找配置文件。

---

## 9. Accelerator 对象:构造参数 + 属性 + 方法全表

### 9.1 构造参数
```python
from accelerate import Accelerator
accelerator = Accelerator(
    mixed_precision="bf16",              # no/fp16/bf16(也可由 launch 控制)
    gradient_accumulation_steps=4,       # 梯度累积
    log_with="wandb",                    # 实验记录后端:wandb/tensorboard(接 wandb 用它)
    project_dir="output",                # 日志/checkpoint 根目录
    step_scheduler_with_optimizer=True,  # 调度器是否跟着优化器步进(默认True)
)
```
| 参数 | 作用 |
|---|---|
| `mixed_precision` | 混合精度模式 |
| `gradient_accumulation_steps` | 累积几步更新一次 |
| `log_with` | tracker 后端(接 wandb) |
| `project_dir` | 日志/状态根目录 |
| `device_placement` | 是否自动把张量放设备(默认 True) |
| `split_batches` | batch 是"每卡各拿 bs"还是"总 bs 拆给各卡"(默认 False=各拿 bs) |
| `kwargs_handlers` | 传底层 DDP 的额外设置(如 `find_unused_parameters`) |

### 9.2 属性:判断"我是谁、在哪张卡"

| 属性 | 是什么 | 典型用途 |
|---|---|---|
| `accelerator.device` | 当前进程的设备(cuda:0/cuda:1) | 手动建张量时 `.to(device)` |
| `accelerator.num_processes` | 总进程数(=卡数) | 算总 batch、总步数 |
| `accelerator.process_index` | 全局进程号(0,1,2...) | = rank |
| `accelerator.local_process_index` | 本机内进程号 | 多机时用 |
| `accelerator.is_main_process` | 是不是全局 0 号 | **只做一次的事**(存模型/传wandb) |
| `accelerator.is_local_main_process` | 是不是本机 0 号 | 每台机器做一次(如下数据) |
| `accelerator.sync_gradients` | 这步是否真更新了参数 | 梯度累积时判断"要不要 log/裁剪" |

### 9.3 方法:核心动作

| 方法 | 干嘛 | 代替了单卡的什么 |
|---|---|---|
| `prepare(model, optim, loader, sched)` | 托管:搬设备+包多卡+切数据 | `.to(device)` + DDP + Sampler |
| `backward(loss)` | 反向(处理混合精度/累积) | `loss.backward()` |
| `clip_grad_norm_(params, max)` | 梯度裁剪(多卡安全版) | `torch.nn.utils.clip_grad_norm_` |
| `accumulate(model)` | 梯度累积上下文 | 手写 no_sync + 计数 |
| `gather(tensor)` | 收集所有卡的张量 | —— |
| `gather_for_metrics(tensor)` | 收集(自动去掉补齐的重复样本) | 算 eval 指标专用 |
| `reduce(tensor, "mean")` | 跨卡归约(求和/平均) | all-reduce |
| `unwrap_model(model)` | 脱掉多卡外壳取原模型 | `model.module` |
| `wait_for_everyone()` | 路障:等所有卡到齐 | `dist.barrier()` |
| `print(...)` | 只主进程打印 | `if rank==0: print` |
| `save(obj, path)` | 只主进程存 | `torch.save` |
| `save_state(dir)` / `load_state(dir)` | 存/读整个训练状态(断点续训) | 手写 ckpt |
| `init_trackers/log/end_training` | 接 wandb(第三部分讲) | —— |
| `get_tracker("wandb")` | 拿底层 wandb run(给 watch 用) | —— |

---

## 10. 实战配方(Cookbook · 10 个)

**每个都能直接抄进脚本。**

### 配方 1:最小多卡训练模板
```python
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

model.train()
for epoch in range(epochs):
    for batch in loader:
        optimizer.zero_grad()
        loss = model(**batch).loss
        accelerator.backward(loss)
        optimizer.step()
    accelerator.print(f"epoch {epoch} loss={loss.item():.4f}")
```

### 配方 2:梯度累积(显存不够,攒着更新)
```python
accelerator = Accelerator(gradient_accumulation_steps=4)
for batch in loader:
    with accelerator.accumulate(model):        # 包住,其余照常写
        loss = model(**batch).loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    if accelerator.sync_gradients:             # 攒够4步、真更新了才做
        global_step += 1
        # ...log / 学习率调度 ...
```

### 配方 3:混合精度 bf16
```python
# 法一(推荐):启动时 --mixed_precision bf16
# 法二:代码里写死
accelerator = Accelerator(mixed_precision="bf16")
# forward 会自动在 bf16 下算,你不用手动 autocast
```

### 配方 4:梯度裁剪(注意"只在真更新的步做")
```python
with accelerator.accumulate(model):
    loss = model(**batch).loss
    accelerator.backward(loss)
    if accelerator.sync_gradients:                       # 关键:累积没满不裁
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); optimizer.zero_grad()
```

### 配方 5:只主进程存模型(标准三连,记死)
```python
accelerator.wait_for_everyone()                          # ① 等所有卡到齐
if accelerator.is_main_process:                          # ② 只主进程
    unwrapped = accelerator.unwrap_model(model)          # ③ 脱壳再存
    unwrapped.save_pretrained("output", save_function=accelerator.save)
    tokenizer.save_pretrained("output")
```

### 配方 6:断点续训(存/读整个状态)
```python
accelerator.save_state("ckpt/step1000")   # 模型+优化器+调度器+step 全存
accelerator.load_state("ckpt/step1000")   # 恢复
# 想让自定义变量(如 global_step)也被存:accelerator.register_for_checkpointing(obj)
```

### 配方 7:分布式下正确算 eval 指标
```python
model.eval()
all_preds, all_labels = [], []
for batch in eval_loader:
    with torch.no_grad():
        preds = model(**batch).logits.argmax(-1)
    # gather_for_metrics:自动去掉"为凑整补的重复样本",指标才准
    preds, labels = accelerator.gather_for_metrics((preds, batch["labels"]))
    all_preds.append(preds); all_labels.append(labels)
# 拼起来算准确率...
```

### 配方 8:只让主进程先做某事(下载/预处理数据)
```python
with accelerator.main_process_first():     # 主进程先跑(下载/缓存),其余等它完再进
    dataset = load_and_cache_dataset()
```

### 配方 9:固定随机种子(多卡可复现)
```python
from accelerate.utils import set_seed
set_seed(42)
```

### 配方 10:DataLoader 注意点
```python
loader = DataLoader(ds, batch_size=4, shuffle=True,
                    num_workers=4, pin_memory=True,   # 提速:多进程读数据 + 锁页内存
                    collate_fn=my_collate)
# ⚠️ shuffle=True 即可,prepare 会自动换成分布式 sampler;别自己加 DistributedSampler(会重复切)
```

---

## 11. Accelerate 排错

| 现象 | 原因 | 解决 |
|---|---|---|
| 每卡都跑全量数据(没加速) | 忘了 prepare(loader) | loader 一定一起 prepare |
| 日志刷 N 份 | 用了 `print` | 换 `accelerator.print` |
| 存的权重名多 `module.` 前缀 | 存了没脱壳的 model | 存前 `unwrap_model` |
| `address already in use` | 端口被占 | `--main_process_port 29501` |
| 卡在开头不动 | 各进程没同步/NCCL 问题 | 检查 `--num_processes` 与实际卡数一致 |
| `find_unused_parameters` 报错 | 模型有分支没参与 loss | 传 kwargs_handlers 开启该选项 |
| 存模型时有的卡没写完 | 没等齐 | 存前 `wait_for_everyone()` |
| `libgomp: Invalid value for OMP_NUM_THREADS` | 环境变量非法 | `export OMP_NUM_THREADS=8` |

---

# 第三部分 · 两个框架合起来

## 12. 标准组合:HF 模型 → Accelerate 多卡训练

**这就是 `train_sft.py` 的骨架。** 看每一步分别用了哪个框架:

```python
# ── HuggingFace:加载三大件 ──
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")            # HF
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)  # HF

# ── HuggingFace:数据 → token(apply_chat_template + labels 掩码)──
#    (在 Dataset.__getitem__ 里,见 train_sft.py)

# ── Accelerate:建总管 + 托管 ──
accelerator = Accelerator()                                       # Accelerate
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)  # Accelerate

# ── 训练循环:HF 出 loss,Accelerate 管多卡 ──
for batch in loader:
    out = model(**batch)             # HF:传 labels 自动算 loss
    accelerator.backward(out.loss)   # Accelerate:多卡反向
    optimizer.step(); optimizer.zero_grad()

# ── 存:Accelerate 脱壳 + HF 保存 ──
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.unwrap_model(model).save_pretrained("output")     # Accelerate 脱壳 + HF 存
```

**分工记忆**:
- **HuggingFace 负责"模型和数据"**:加载、分词、算 loss、生成、保存。
- **Accelerate 负责"把训练铺到多卡"**:托管、反向、汇总、只主进程存。
- 两者接口干净:HF 的 `model` 直接丢给 Accelerate 的 `prepare`,无缝。

## 13. 完整可复制模板

```python
import math, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

def main():
    set_seed(42)
    accelerator = Accelerator()                                          # Accelerate

    # —— HuggingFace:模型 + 分词器 ——
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.config.use_cache = False

    # —— 数据(你的 Dataset/collate,用 apply_chat_template + -100 掩码)——
    loader = DataLoader(MyDataset(tok), batch_size=4, shuffle=True, collate_fn=my_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # —— Accelerate:托管 ——
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    total = math.ceil(len(loader)) * 3
    sched = get_cosine_schedule_with_warmup(optimizer, int(0.03*total), total)

    # —— 训练 ——
    model.train()
    for epoch in range(3):
        for batch in loader:
            out = model(**batch)                     # HF:算 loss
            accelerator.backward(out.loss)           # Accelerate:反向
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched.step(); optimizer.zero_grad()
        accelerator.print(f"epoch {epoch} loss {out.loss.item():.4f}")

    # —— 存 ——
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained("output")
        tok.save_pretrained("output")

if __name__ == "__main__":
    main()
```
跑:`accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 train.py`

---

# 第四部分 · 速查小抄

```text
【HuggingFace transformers】
  分词器   tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
  模型     model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
  编码     enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
  解码     tok.decode(ids, skip_special_tokens=True)
  对话拼   tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False)
  算loss   out = model(input_ids=, attention_mask=, labels=);  loss = out.loss
  生成     model.generate(**inputs, max_new_tokens=200, do_sample=False)
  保存     model.save_pretrained("output");  tok.save_pretrained("output")
  下载     export HF_ENDPOINT=https://hf-mirror.com   (国内镜像)

【datasets】
  ds = load_dataset("shibing624/alpaca-zh", split="train")
  ds.select(range(10000)) / ds.map(fn) / ds.filter(fn) / ds.to_json("x.jsonl", force_ascii=False)

【Accelerate】
  建总管   acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
  托管     model,opt,loader = acc.prepare(model,opt,loader)
  反向     acc.backward(loss)
  累积     with acc.accumulate(model): ...   ;  真更新? if acc.sync_gradients:
  裁剪     acc.clip_grad_norm_(model.parameters(), 1.0)
  汇总     acc.gather(loss).mean()
  只主进程 if acc.is_main_process: / acc.print(...)
  等齐+存  acc.wait_for_everyone(); acc.unwrap_model(model).save_pretrained("out")
  断点     acc.save_state("ckpt") / acc.load_state("ckpt")
  启动     accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 train.py

【分工】 HuggingFace 管"模型和数据"  |  Accelerate 管"铺到多卡"
```
