# SFT 数据集格式 + Loss Function 详解(深入版)

> 任务第②项。目标:彻底搞懂 ① SFT 数据长什么样(各种格式)② loss 怎么从头推出来(那个 $-\log$)。
> 前置:已懂 HF 文件格式(`SFT任务_01`)、GRPO 笔记里的"SFT=抄范文"。
>
> 📐 本文公式用 LaTeX 写,请用**支持数学的 Markdown 阅读器**看(Typora / Obsidian / VSCode 预览 / GitHub),否则会看到 `$$` 源码。

---

## 目录
- [Part A. 数据集格式(各种主流格式)](#part-a-数据集格式各种主流格式)
  - [A0. 先分清:预训练数据 vs SFT 数据](#a0-先分清预训练数据-vs-sft-数据)
  - [A1. 五种主流数据格式(带例子)](#a1-五种主流数据格式带例子)
  - [A2. chat template:messages 怎么变成字符串(ChatML)](#a2-chat-templatemessages-怎么变成字符串chatml)
  - [A3. 多轮对话怎么摆](#a3-多轮对话怎么摆)
  - [A4. 从对话到 input_ids + labels(loss masking 详解)](#a4-从对话到-input_ids--labelsloss-masking-详解)
  - [A5. next-token 的"错位对齐"(shift)](#a5-next-token-的错位对齐shift)
  - [A6. 组 batch:padding + attention_mask](#a6-组-batchpadding--attention_mask)
- [Part B. Loss Function 从头推导](#part-b-loss-function-从头推导)
  - [B0. 一句话目标:最大似然](#b0-一句话目标最大似然)
  - [B1. 第一步:语言模型的概率分解(链式法则)](#b1-第一步语言模型的概率分解链式法则)
  - [B2. 第二步:取 log,最大化变最小化(负号哪来的)](#b2-第二步取-log最大化变最小化负号哪来的)
  - [B3. 第三步:softmax —— logits 怎么变概率](#b3-第三步softmax--logits-怎么变概率)
  - [B4. 第四步:交叉熵 = NLL(one-hot 塌缩)](#b4-第四步交叉熵--nllone-hot-塌缩)
  - [B5. 第五步:展开成代码里真正算的形式(log-sum-exp)](#b5-第五步展开成代码里真正算的形式log-sum-exp)
  - [B6. 第六步:整条序列 + masking 求平均](#b6-第六步整条序列--masking-求平均)
  - [B7. 一个完整的数值例子(手算一遍)](#b7-一个完整的数值例子手算一遍)
  - [B8. 进阶:梯度 = softmax − onehot(为什么好训)](#b8-进阶梯度--softmax--onehot为什么好训)
  - [B9. 数值稳定:为什么用 log_softmax](#b9-数值稳定为什么用-log_softmax)
- [Part C. 串起来 + 自测](#part-c-串起来--自测)

---

# Part A. 数据集格式(各种主流格式)

## A0. 先分清:预训练数据 vs SFT 数据

| | 预训练(pretrain) | 微调(SFT) |
|---|---|---|
| 数据 | 一大堆**无结构的连续文本** | 一条条**「问→答」对话** |
| 目标 | 学语言本身 | 学"照着指令回答" |
| 算 loss 的范围 | **每个 token 都算** | **只有「回答」部分算**(重点!) |

预训练数据长这样(没结构,给什么学什么):
```json
{"text": "水在标准大气压下的沸点是 100 摄氏度。这一现象..."}
```

SFT 数据必须有"谁问、谁答"的结构,下面细讲。

---

## A1. 五种主流数据格式(带例子)

同一件事(问答对),不同项目/框架用不同的 JSON 字段名。你会遇到这几种,**认得出、能互转**即可:

### ① Alpaca 格式(单轮指令,最经典)
```json
{
  "instruction": "把下面的句子翻译成英文。",
  "input": "今天天气很好。",
  "output": "The weather is nice today."
}
```
- `instruction`(指令)+ `input`(可选的输入内容)拼成**问**;`output` 是**答**。
- `input` 可以为空(比如"写一首诗"就不需要 input)。
- 谁在用:Stanford Alpaca、很多早期开源 SFT 数据。

### ② ShareGPT / conversations 格式(多轮对话)
```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "gpt",   "value": "你好!有什么可以帮你?"},
    {"from": "human", "value": "讲个笑话"},
    {"from": "gpt",   "value": "为什么程序员分不清万圣节和圣诞节?因为 Oct 31 == Dec 25。"}
  ]
}
```
- 角色字段叫 `from`(值 human/gpt/system),内容字段叫 `value`。
- **支持多轮**(一问一答来回好几次)。
- 谁在用:ShareGPT 数据、Vicuna,以及很多多轮数据集。

### ③ OpenAI messages 格式(现代标准,Qwen 用这个)⭐
```json
{
  "messages": [
    {"role": "system",    "content": "你是一个乐于助人的助手。"},
    {"role": "user",      "content": "1+1等于几?"},
    {"role": "assistant", "content": "等于2。"}
  ]
}
```
- 角色字段叫 `role`(值 system/user/assistant),内容字段叫 `content`。
- `system` = 系统提示(设定人设/规则,可选,通常放最前面一条)。
- **这是当前最主流的格式**,也是 Qwen3 的 chat template 直接吃的格式。**你的任务优先用这个。**

### ④ prompt-completion 格式(最朴素)
```json
{"prompt": "1+1=", "completion": "2"}
```
- 直接给"提示"和"补全"。TRL 的一些接口支持。适合简单任务。

### ⑤ (对照)raw text 格式 —— 这是预训练,不是 SFT
```json
{"text": "任意一段连续文本……"}
```
- 无角色、无 masking,每个 token 都算 loss。列在这里是让你对比,别搞混。

> **小结**:①②③④ 都是"问答对",只是字段名不同。**本质都会被转成 messages,再走 chat template。** 记住主流是 ③ messages。

---

## A2. chat template:messages 怎么变成字符串(ChatML)

模型不能直接吃 JSON,得先把 messages 拼成**一个字符串**。Qwen 用的格式叫 **ChatML**,长这样:

```
<|im_start|>system
你是一个乐于助人的助手。<|im_end|>
<|im_start|>user
1+1等于几?<|im_end|>
<|im_start|>assistant
等于2。<|im_end|>
```

拆解:
- `<|im_start|>` / `<|im_end|>`:特殊 token,标记每段对话的**开始/结束**(im = instant message)。
- 每段格式:`<|im_start|>` + 角色名 + 换行 + 内容 + `<|im_end|>`。
- 最后那个 `<|im_end|>` 很重要——**训练时要让模型学会答完生成它来"停下"**。

代码里一行搞定(你在 `inspect_qwen.py` 第4步见过):
```python
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

> 每个模型的 template 不同(Llama、ChatGLM 各有各的特殊 token)。**别自己手拼,一定用 `apply_chat_template`**,它读的是 `tokenizer_config.json` 里定义的模板,保证和模型训练时一致。

---

## A3. 多轮对话怎么摆

多轮就是 messages 里多几组 user/assistant:
```
<|im_start|>system\n系统提示<|im_end|>
<|im_start|>user\n问题1<|im_end|>
<|im_start|>assistant\n回答1<|im_end|>      ← 这段算 loss
<|im_start|>user\n问题2<|im_end|>
<|im_start|>assistant\n回答2<|im_end|>      ← 这段也算 loss
```
- **所有 assistant 回答都算 loss**,所有 system/user 都不算。
- 一条多轮数据 = 一个训练样本(整段拼一起喂进去)。

---

## A4. 从对话到 input_ids + labels(loss masking 详解)

**这是 SFT 的灵魂,单独拎出来讲透。**

拼好字符串后,tokenizer 把它切成一串 token id,叫 `input_ids`。同时要造一个等长的 `labels`,决定**哪些位置算 loss**:

- **算 loss 的位置**:labels 填**真实的 token id**。
- **不算 loss 的位置**:labels 填 **`-100`**(PyTorch 交叉熵的约定:label=-100 的位置**直接跳过、不算 loss、不回传梯度**)。

规则:**只有「assistant 的回答内容 + 它结尾的 `<|im_end|>`」保留真实 id,其余(system、user、以及 assistant 的头 `<|im_start|>assistant\n`)全设 -100。**

具体到一个例子(简化,一个中文字≈一个token,示意):

| 位置 | token | input_ids | labels | 算loss? |
|---|---|---|---|---|
| 0 | `<\|im_start\|>` | 151644 | **-100** | ✗ 属于prompt |
| 1 | `user` | ... | **-100** | ✗ |
| 2 | `\n` | ... | **-100** | ✗ |
| 3 | `1+1等于几?` | ... | **-100** | ✗ 用户的问题 |
| 4 | `<\|im_end\|>` | 151645 | **-100** | ✗ |
| 5 | `<\|im_start\|>` | 151644 | **-100** | ✗ assistant头也不算 |
| 6 | `assistant` | ... | **-100** | ✗ |
| 7 | `\n` | ... | **-100** | ✗ |
| 8 | `等于2。` | 真实id | **真实id** | ✅ 回答内容 |
| 9 | `<\|im_end\|>` | 151645 | **151645** | ✅ 让它学会停下 |

> 🔑 记死:**SFT = 只在「assistant 回答 + 结尾 im_end」上算 loss,其它全 -100。** 为什么?你要模型学"**怎么答**",不是学"怎么问"——问题是给定的输入,不需要模型去生成。

> 顺带:为什么 assistant 的头 `<|im_start|>assistant\n` 也设 -100?因为那是**格式脚手架**,是提示模型"该你答了"的固定前缀,不是模型要学着生成的内容。学它没意义。

---

## A5. next-token 的"错位对齐"(shift)

有个容易懵的细节:模型在位置 $t$ 看到的是 $x_1,\dots,x_t$,要预测的是**下一个** $x_{t+1}$。所以 logits 和 labels 之间差一位。

HF 的处理方式(你只要知道结论):
- 你喂给模型 `input_ids` 和**等长**的 `labels`(位置一一对应)。
- 模型**内部自动错位一格**再算 loss,代码里是:
```python
shift_logits = logits[:, :-1, :]   # 去掉最后一个位置的预测
shift_labels = labels[:, 1:]       # 去掉第一个位置的标签
loss = cross_entropy(shift_logits, shift_labels)
```
- 意思:**用第 $t$ 位的预测,去对第 $t+1$ 位的真实 token**。

> 结论:**你不用自己手动错位**。你只要保证 `labels` 和 `input_ids` **同长、位置对齐**,该 mask 的地方填 -100,shift 交给模型。

---

## A6. 组 batch:padding + attention_mask

一个 batch 里每条对话长短不一,要对齐:
- **padding**:短的用 `pad_token` 补到和最长的一样长。
- **attention_mask**:标记哪些是真 token(1)、哪些是 padding(0),让模型**注意力忽略 padding**。
- **labels 的 padding 位置也填 -100**(padding 不该算 loss)。

```
input_ids:      [151644, ...,  等于2, 151645, PAD,  PAD ]
attention_mask: [   1,   ...,   1,     1,     0,    0  ]
labels:         [ -100,  ...,  等于2, 151645, -100, -100]
```

---

# Part B. Loss Function 从头推导

**目标:把那个 $-\log$ 从最根上推出来,不跳步。** 跟着走一遍,你就再也不会觉得它是"背的公式"。

## B0. 一句话目标:最大似然

我们想让模型**给真实数据打高概率**。也就是:训练数据里那句真实的回答,模型觉得它"越可能出现越好"。这个原则叫**最大似然(Maximum Likelihood)**。整个 loss 就是从这一句话推出来的。

## B1. 第一步:语言模型的概率分解(链式法则)

一句话是一串 token:$x = (x_1, x_2, \dots, x_T)$。这句话整体的概率,用**概率的链式法则**(恒等式,不是近似)拆开:

$$
P(x_1, x_2, \dots, x_T)
= P(x_1)\,P(x_2\mid x_1)\,P(x_3\mid x_1,x_2)\cdots P(x_T\mid x_1,\dots,x_{T-1})
= \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

其中 $x_{<t}$ = "位置 $t$ 之前的所有 token"。

> 🔑 **这一步解释了为什么大模型是"预测下一个词"**:建模整句的概率,等价于反复建模 `P(下一个词 | 前面所有词)`。这就是"自回归 / next-token prediction"的数学来源。也是你 VAR 里"next-scale"、GPT 里"next-token"共享的底层思想。

模型带参数 $\theta$,写成 $P_\theta(x_t \mid x_{<t})$。

## B2. 第二步:取 log,最大化变最小化(负号哪来的)

我们要在整个数据集上**最大化**上面那个连乘。但**连乘有两个问题**:①很多小于1的数连乘会下溢到0;②求导难。**取对数**解决(log 是单调增,最大化 $\log$ 等价于最大化原式,不改变最优解):

$$
\log P_\theta(x) = \log \prod_{t=1}^{T} P_\theta(x_t\mid x_{<t}) = \sum_{t=1}^{T} \log P_\theta(x_t\mid x_{<t})
$$

连乘变**连加**,舒服多了。这个 $\log P_\theta(x)$ 叫**对数似然(log-likelihood)**。

现在:我们想**最大化**对数似然。但优化器(SGD/Adam)的惯例是**最小化**。于是**加个负号**,"最大化对数似然" = "最小化负对数似然(Negative Log-Likelihood, NLL)":

$$
\text{Loss} = -\log P_\theta(x) = \sum_{t=1}^{T}\bigl[-\log P_\theta(x_t\mid x_{<t})\bigr]
$$

> 🔑 **$-\log$ 里的负号,就是从这一步来的**——不是随便加的,是"把最大化翻译成最小化"的结果。到这里,loss 已经成型:**每个位置的损失 = $-\log(\text{模型给正确下一个词的概率})$,全部加起来。**

## B3. 第三步:softmax —— logits 怎么变概率

上面的 $P_\theta(x_t \mid x_{<t})$ 具体怎么算出来?模型在位置 $t$ 会吐出一个长度=词表大小 $V$(Qwen 是 151936)的向量 $z = (z_1,\dots,z_V)$,叫 **logits**(每个词的"原始分数",可正可负,不是概率)。

用 **softmax** 把 logits 变成合法概率(非负、和为1):

$$
P_\theta(x_t = i \mid x_{<t}) = \operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}
$$

- $\exp$ 保证非负;除以总和保证加起来=1。
- 分子 $\exp(z_i)$ 越大(该词 logit 越高),这个词概率越大。

设这一步的**正确词**下标是 $c$(真实的下一个 token),那么模型给正确词的概率是:

$$
p_c = \frac{\exp(z_c)}{\sum_{j=1}^{V}\exp(z_j)}
$$

## B4. 第四步:交叉熵 = NLL(one-hot 塌缩)

你常听到 SFT 用"**交叉熵损失(Cross-Entropy)**"。这里说明它**就是** B2 推出的 NLL,不是另一个东西。

交叉熵的通用定义(两个分布 $q$ 真实、$p$ 预测):

$$
H(q, p) = -\sum_{i=1}^{V} q_i \log p_i
$$

在语言模型里,"真实的下一个词"是**确定的**——就是某个具体词 $c$。所以真实分布 $q$ 是 **one-hot**(独热):$q_c = 1$,其余 $q_i = 0$。代入:

$$
\begin{aligned}
H(q, p) &= -\sum_i q_i \log p_i \\
        &= -\Bigl( q_c\log p_c + \sum_{i\ne c} q_i\log p_i \Bigr) \\
        &= -\Bigl( 1\cdot\log p_c + \sum_{i\ne c} 0\cdot\log p_i \Bigr) \\
        &= -\log p_c
\end{aligned}
$$

> 🔑 **交叉熵 + one-hot 目标,塌缩成 $-\log(\text{正确词概率})$。** 这就把"交叉熵"这个名字和"$-\log p$"这个公式对上了——**它俩是一个东西**。所以说 SFT loss = 交叉熵 = NLL = $-\log p_c$。

## B5. 第五步:展开成代码里真正算的形式(log-sum-exp)

把 B3 的 $p_c$ 代进 $-\log p_c$,展开:

$$
\begin{aligned}
\text{loss}_t &= -\log p_c \\
              &= -\log \frac{\exp(z_c)}{\sum_j \exp(z_j)} \\
              &= -\bigl[\, \log\exp(z_c) - \log\textstyle\sum_j \exp(z_j) \,\bigr] \\
              &= -\bigl[\, z_c - \log\textstyle\sum_j \exp(z_j) \,\bigr] \\
              &= -z_c + \log\sum_{j=1}^{V}\exp(z_j)
\end{aligned}
$$

> 上面用到两条对数性质:第 3 行 $\log\frac{a}{b}=\log a-\log b$(除法变相减);第 4 行 $\log\exp(z_c)=z_c$(log 和 exp 抵消)。

即:

$$
\text{loss}_t = -z_c + \operatorname{logsumexp}(z)
$$

- $z_c$ = 正确词的 logit;$\operatorname{logsumexp}(z) = \log\sum_j \exp(z_j)$。
- **这就是 `F.cross_entropy` 内部真正算的东西**——直接吃 logits(不用你先手动 softmax)。

## B6. 第六步:整条序列 + masking 求平均

一条数据有很多 token,但 SFT **只在没被 mask 的位置(labels $\ne -100$,即 assistant 回答)算**。设这些位置的集合是 $M$:

$$
\begin{aligned}
\text{Loss} &= \frac{1}{|M|}\sum_{t\in M}\bigl[-\log P_\theta(x_t\mid x_{<t})\bigr] \\
            &= \frac{1}{|M|}\sum_{t\in M}\bigl[-z_{t,\,c_t} + \operatorname{logsumexp}(z_t)\bigr]
\end{aligned}
$$

- $|M|$ = 回答部分的 token 数(被 -100 屏蔽的不进分母)。
- 对这些位置的 $-\log p$ **求平均**,就是这条数据最终的 loss。

> 到此,从"最大似然"一路推到了代码里那个 `cross_entropy(logits, labels, ignore_index=-100)`。**每一块你都知道为什么了。**

## B7. 一个完整的数值例子(手算一遍)

假设词表只有 5 个词,某个位置模型输出 logits:

$$
z = [\,2.0,\ 1.0,\ 0.1,\ 0.0,\ -1.0\,]
$$

**Step1 求 $\exp$:**

$$
\exp(z) = [\,7.389,\ 2.718,\ 1.105,\ 1.000,\ 0.368\,],\qquad \sum_j \exp(z_j) = 12.580
$$

**Step2 softmax(每个词的概率):**

$$
p = [\,0.587,\ 0.216,\ 0.088,\ 0.079,\ 0.029\,]
$$

(每个 $p_i = \exp(z_i)/12.580$,加起来 $=1$ ✓)

**情况A:正确词是第 0 个($p_0 = 0.587$):**

$$
\text{loss} = -\log(0.587) = 0.533
$$

**用 B5 的 log-sum-exp 形式验证(应相等):**

$$
\text{loss} = -z_0 + \log\textstyle\sum_j\exp(z_j) = -2.0 + \log(12.580) = -2.0 + 2.532 = 0.532 \ \checkmark
$$

**情况B:正确词是第 4 个($p_4 = 0.029$,模型几乎没想到):**

$$
\text{loss} = -\log(0.029) = 3.53
$$

→ 比情况 A 大得多,**狠狠惩罚**。

> 直觉印证:模型对正确词**越有把握($p$ 越大),loss 越小;越没料到,loss 越大**。这就是 $-\log$ 当"惩罚尺子"的意义:

| 模型给正确词的概率 $p$ | $-\log(p)$ | 含义 |
|---|---|---|
| 1.0(完全确定,答对) | 0 | 不罚 |
| 0.5 | 0.69 | 小罚 |
| 0.1 | 2.30 | 中罚 |
| 0.01 | 4.61 | 重罚 |
| $\to 0$(完全没料到) | $\to\infty$ | 罚到爆 |

## B8. 进阶:梯度 = softmax − onehot(为什么好训)

交叉熵对 logits 的梯度特别干净(可自己推,或先记结论):

$$
\frac{\partial\,\text{loss}_t}{\partial z_i} = \operatorname{softmax}(z)_i - q_i = p_i - \mathbf{1}[\,i = c\,]
$$

- 对**正确词** $c$:梯度 $= p_c - 1$($p_c<1$ 时为负)→ 更新会**推高** $z_c$ → 正确词概率↑。
- 对**错误词** $i$:梯度 $= p_i - 0 = p_i$(正)→ 更新会**压低** $z_i$ → 错误词概率↓。
- 力度 $\propto$ "你给错的概率有多离谱"。

> 🔑 **"推高正确词、压低错误词",这和你 GRPO 笔记里"把好答案概率往上推、坏的往下压"是同一个动作**——只不过 SFT 的"好"是"数据里的标准答案",GRPO 的"好"是"reward 比组平均高"。**底层都是在动概率。**

## B9. 数值稳定:为什么用 log_softmax

直接"先 softmax 再 log"会出事:logit 很大时 $\exp$ 溢出;$p$ 接近 0 时 $\log$ 变 $-\infty$。所以框架**直接算 log_softmax**,并用"减最大值"技巧:

$$
\operatorname{logsumexp}(z) = m + \log\sum_j \exp(z_j - m),\qquad m = \max(z)
$$

减掉 $m$ 后 $\exp$ 的输入 $\le 0$,不会溢出,结果不变。

> 实践结论:**永远把 raw logits 交给 `F.cross_entropy` / `nn.CrossEntropyLoss`,别自己先 softmax**。它内部用 log_softmax,又稳又对。

---

# Part C. 串起来 + 自测

**SFT 完整流程(把 Part A + B 接起来):**
```
1. 数据:一堆 messages(system/user/assistant)
2. apply_chat_template → 拼成 ChatML 字符串
3. tokenize → input_ids;造 labels:回答部分留真实id,其余填 -100
4. 组 batch:padding + attention_mask + labels补-100
5. 前向:模型每个位置输出 V 维 logits
6. 算 loss:只在 labels≠-100 的位置,loss = 平均的 −z_c + logsumexp(z)(=交叉熵)
7. backward + step:推高正确词、压低错误词 → 模型越来越会答
```

**代码骨架(核心就这几行):**
```python
out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = out.loss          # HF 内部:shift + cross_entropy(ignore_index=-100),就是 Part B
loss.backward()
optimizer.step()
```

**自测题(能用大白话+公式答就过):**
1. SFT 和预训练,算 loss 的范围有什么根本区别?
2. 列出至少 3 种数据格式的字段名(Alpaca / ShareGPT / messages)。
3. `-100` 是干嘛的?为什么 user 问题部分要设 -100?
4. **从最大似然推到 $-\log$:链式法则 → 取 log → 加负号,负号是怎么来的?**⭐
5. 交叉熵 $H(q,p)=-\sum_i q_i \log p_i$ 在 one-hot 目标下为什么塌缩成 $-\log p_c$?
6. $\text{loss}_t = -z_c + \operatorname{logsumexp}(z)$ 是怎么从 $-\log p_c$ 展开的?
7. 模型给正确词概率 0.9,loss 约多少?给 0.1 呢?(手算)
8. 交叉熵对 logits 的梯度是什么?它和 GRPO"推高/压低概率"有什么共通?
