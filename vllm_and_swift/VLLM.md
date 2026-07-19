# vLLM 从零详解 · 高吞吐 LLM 推理引擎(主讲稿底稿)

> 师兄本周任务:VLLM & SWIFT 体验、ReAct 论文、Tool Use RL。你分工 = **主讲 vLLM + ReAct**。
> 这份是 vLLM 那半的底稿。任务里明确要 **"对比 HF 推理和 vLLM 推理速度差异(Qwen3-3B),理解原因"** ——
> 所以这份文档 60% 的篇幅在讲 **"为什么快"**,因为上台被问到的一定是这个,不是"怎么装"。
>
> 读法:Part 0 看类比 → Part 1~3 是**地基**(LLM 怎么推理 + 为什么慢),这三节最重要,想通了后面都是水到渠成 →
> Part 4~7 是 vLLM 的**两张王牌 + 因果链** → Part 8~11 上手 + 师兄任务①实战脚本 → 剩下当手册查。
> 你已经会的东西(AR 自回归、KV、DDP 数据并行)我都会拿来搭桥,不重复造轮子。

---

## 目录
- [Part 0. 一句话 + 为什么需要它(食堂打饭类比)](#part-0)
- [Part 1. 地基:LLM 到底怎么"推理"的(预填充 + 逐字解码)](#part-1)
- [Part 2. KV Cache:推理的命根子(为什么占显存)](#part-2)
- [Part 3. 解码为什么慢?——是"内存墙",不是"算力不够"⭐最重要](#part-3)
- [Part 4. 传统方式(HF)到底浪费在哪](#part-4)
- [Part 5. vLLM 王牌一:PagedAttention(把 KV 当操作系统内存管)⭐](#part-5)
- [Part 6. vLLM 王牌二:连续批处理(Continuous Batching)⭐](#part-6)
- [Part 7. 合起来:为什么快 10~24 倍(一条因果链讲透)⭐](#part-7)
- [Part 8. 上手:安装 + 两种用法](#part-8)
- [Part 9. 关键参数(调不好就 OOM / 变慢)](#part-9)
- [Part 10. 多卡:张量并行(和你会的 DDP 数据并行区别)](#part-10)
- [Part 11. 师兄任务①实战:HF vs vLLM 测速(Qwen3-3B)⭐](#part-11)
- [Part 12. 什么时候 vLLM 帮助没那么大 + 坑速查](#part-12)
- [Part 13. 和别的推理引擎的关系(TGI / TensorRT-LLM / SGLang)](#part-13)
- [Part 14. 自测(上台前自己过一遍)](#part-14)
- [Part 15. 术语表(最后回填)](#part-15)

---

<a id="part-0"></a>
# Part 0. 一句话 + 为什么需要它(食堂打饭类比)

## 一句话

> **vLLM = 让同一张 GPU 上,LLM 推理的"每秒吐字数(吞吐)"翻 10~24 倍的引擎。**
> 它没换模型、没减精度,靠的是两件事:① 把 KV cache 当**操作系统内存**那样分页管理(不浪费显存);② **连续拼批**(GPU 一刻不空转)。

注意关键词是 **吞吐(throughput)**,不是**单条快**。vLLM 的主场是"很多人同时请求"(服务化),不是"你一个人问一句"。这点后面 Part 12 会强调,面试/主讲被追问时别答错。

## 痛点:你已经体会过的那个"慢"

你在 `chat_compare.py` 里用 `model.generate()` 一句一句问 Qwen,是不是觉得**一个字一个字往外蹦、很慢**?那就是 HuggingFace transformers 的原生推理。它:
- 一次只顺畅地服务少数几条请求;
- 显存大量浪费在"预留但没用上"的 KV 空间;
- 短请求要排在长请求后面干等。

**vLLM 就是专门解决"把一个模型部署成服务、同时扛很多请求"这件事的**。它是 UC Berkeley 2023 年的论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》(SOSP'23)配套的开源引擎,现在是业界部署开源大模型**事实上的标准**(几乎所有"本地起个 OpenAI 接口"的教程都用它)。

## 类比:食堂打饭 🍚

把"GPU 跑一次模型权重"想成"**食堂开一次打饭窗口**":

| 场景 | 传统 HF transformers | vLLM |
|---|---|---|
| 打饭方式 | 一次只给 1 个人打完整套餐,下一个人排队 | 窗口开一次,**一排人同时把饭盆递上来**,一勺过去每人都打到 |
| 餐盘发放 | 每人**预定一个能装满汉堡套餐的大托盘**,哪怕他只打了二两饭 → 托盘不够用,后面人没盘子 | 饭菜按**固定小格**发,打多少给多少格,格子用完还回收 |
| 有人打完了 | 得等**这一整批人**全打完,才放下一批进来 | 谁打完谁走,**门口的人立刻补位**,队伍始终是满的 |

- "一排人同时打" = **连续批处理**(Part 6);
- "按小格发餐盘、不预定大托盘" = **PagedAttention**(Part 5);
- 这两招合起来,让**同一个打饭窗口(同一张 GPU)单位时间喂饱的人数**暴涨。这就是吞吐翻 10 倍的来源。

记住这个食堂,后面全程用它。

---

<a id="part-1"></a>
# Part 1. 地基:LLM 到底怎么"推理"的(预填充 + 逐字解码)⭐

> 这一节和下一节是**整份文档的地基**。不懂 LLM 推理的两阶段,后面 vLLM 的所有优化都会像空中楼阁。慢点看。

## 先接上你已经会的:训练 vs 推理

你在 SFT / VAR 里学的是**训练**:teacher forcing,把整句话一次性喂进去,并行算所有位置的 loss(还记得那个 `-100` 掩码、`out.loss` 内部帮你 shift 吗)。

**推理不一样**:你没有"标准答案"可喂,模型得**真的一个 token 一个 token 自己往外蹦**(autoregressive,自回归)——这正是 AR 模型的定义。生成第 N 个 token 时,要"看着"前面已经生成的 N-1 个 token。

## 推理分两个阶段(务必记牢)

一次完整的"你问一句,模型答一段",在底层分成**两个性质完全不同**的阶段:

### 阶段一:预填充(Prefill)—— 处理你的问题(prompt)

- 你输入 `"用一句话解释什么是注意力机制"`,假设是 12 个 token。
- 模型把这 **12 个 token 一次性、并行地**全部算一遍(就像训练时那样,一股脑喂进去)。
- 产出:①每个 token 的 K、V 向量(存进 KV cache,见 Part 2);②**第 1 个**输出 token(比如 "注意力")。
- 特点:**一次处理很多 token,GPU 算力吃得很满**(算得多,搬权重的开销被摊薄)→ **算力密集(compute-bound)**。

### 阶段二:解码(Decode)—— 一个字一个字往外蹦

- 拿着刚生成的 "注意力",算出下一个 token "是";再拿 "是" 算出 "让";再算 "模型"……
- **每一步只处理 1 个 token**,处理完立刻要用它算下一个,**天生串行,躲不掉**。
- 每一步都要"回看"前面所有 token(靠 KV cache,不用重算,见 Part 2)。
- 生成 200 个 token,就要跑 **200 个这样的串行小步**。
- 特点:**每步只算 1 个 token 的量,GPU 算力大量闲置,瓶颈变成"把模型权重从显存搬进计算核心"** → **内存带宽密集(memory-bound)**。⭐这句是 Part 3 的引子,划出来。

## 一张时间线看懂

```
输入: "用一句话解释注意力"(12 token)
                                 输出: "注意力 是 让 模型 关注 重点 的 机制"(8 token)

时间 →
┌─────────── Prefill ───────────┐┌─Decode─┐┌─Decode─┐┌─Decode─┐ ... (共 8 步)
│  12 个输入 token 一次并行算完   ││ 出"是" ││ 出"让" ││出"模型"│
│  顺带吐出第 1 个 token "注意力"  ││        ││        ││        │
└───────────────────────────────┘└────────┘└────────┘└────────┘
   算力吃满(compute-bound)          每步只算1个token(memory-bound,慢)
```

**一句话总结 Part 1**:LLM 推理 = 1 次并行的预填充(啃完你的问题)+ N 次串行的解码(一个字一个字吐答案)。**大部分时间花在又慢又串行的解码上**,而解码是"内存墙"问题——这是理解 vLLM 的钥匙。

---

<a id="part-2"></a>
# Part 2. KV Cache:推理的命根子(为什么占显存)⭐

## 为什么需要缓存

注意力机制里,生成一个新 token,要拿它的 Query 去和**前面所有 token 的 Key、Value**做注意力。

- **不缓存**:生成第 100 个 token 时,把前 99 个 token 的 K、V **重新算一遍**;生成第 101 个,再把前 100 个重算……总计算量 ~ O(N²),巨慢且重复。
- **缓存(KV Cache)**:前面 token 的 K、V 算过一次就**存起来**。生成新 token 时,只算**它自己**的 Q、K、V,然后 K、V 追加进缓存。每步只多算 1 个 token → 省掉海量重算。

> 类比:你读一本书边读边做笔记(K、V)。读到第 100 页想回顾前情,直接翻笔记(缓存),而不是把前 99 页**重读一遍**(重算)。

**所以 KV Cache 是"用显存换计算"**——它是解码能快起来的前提,但代价是它**吃显存,而且吃得很凶**。

## KV Cache 有多大?(建立数感,主讲会被问)

每生成/处理一个 token,每一层 Transformer 都要存它的 K 和 V 两个向量。总大小公式:

```
KV显存 = 2(K和V) × 层数 × KV头数 × head_dim × 序列长度 × 并发条数 × 每个数的字节数
         ↑                ↑ 用 KV 头数,不是注意力头数(见下 GQA)              ↑ bf16=2字节
```

**举个例(数字仅为示意,以你下载的 `config.json` 为准)**:一个约 3B、36 层、KV 头 8、head_dim 128、bf16 的模型:

- **每个 token** 的 KV = 2 × 36 × 8 × 128 × 2 字节 ≈ **144 KB**
- 一条 2048 长度的序列 ≈ 144 KB × 2048 ≈ **294 MB**
- 想同时服务 **50 条**这样的序列 ≈ **14 GB** 显存 —— 这还只是 KV,模型权重(3B×2字节≈6GB)另算!

**结论**:一张 24GB 的 4090,装完权重后剩下的显存,**几乎全被 KV cache 吃掉,而它直接决定你能同时塞下多少条请求(batch 能多大)**。记住这句,Part 3、Part 5 全靠它。

## 顺带一个名词:GQA(为什么上面用"KV 头数")

现代模型(Qwen3、Llama3 等)用 **GQA(分组查询注意力)**:让多个 Query 头**共享**同一组 K、V 头。比如 16 个 Q 头只配 8 个 KV 头 → KV cache 直接砍一半。这就是为什么公式里是"KV 头数"而不是"注意力头数"。你主讲时提一句"Qwen3 用 GQA,所以 KV 缓存本身已经被压过一道",显得懂。

---

<a id="part-3"></a>
# Part 3. 解码为什么慢?——是"内存墙",不是"算力不够" 

## 一个反直觉的事实

你可能以为 GPU 慢是因为"算不过来"。**解码阶段恰恰相反:GPU 的计算单元大部分时间在发呆,真正的瓶颈是"把模型权重从显存(HBM)搬到计算核心"。**

为什么?回忆 Part 1:**解码每步只处理 1 个 token**。

- 为了给这 1 个 token 算一层,你得把这一层的**全部权重**从显存读进来。
- 读进来的这一大堆权重,只用来算**1 个 token**的一点点乘加,然后就扔了,下一步还得重读。
- 好比:你**开一趟大卡车(搬全部权重)到工地,只卸下一块砖(算1个token)就开回去**。卡车的运力(算力)完全浪费在"跑空趟"上,瓶颈是**来回搬运(内存带宽)**,不是"搬得动多少"(算力)。

这就是所谓 **"内存墙 / memory-bound"**:限制你的是显存带宽,不是 FLOPs。

## 关键推论:批量(batch)是免费的午餐 🍟

既然"搬一趟权重"这么贵、而搬来的权重又没用满,那——

> **一趟卡车,顺路多拉几块砖不就赚了?**
> 把权重从显存搬进来这一次,**同时给一批(batch)序列各算它们的那 1 个 token**。权重只搬一次,却服务了一整批!

- 处理 batch=1:搬一次权重,产出 1 个 token。
- 处理 batch=32:搬**同样一次**权重,产出 32 个 token,耗时几乎没变。
- → **吞吐(每秒总 token 数)几乎随 batch 线性增长**,直到把算力或显存吃满为止。

**这就是整个高吞吐推理的核心游戏规则:**

> ### 🎯 把 batch 做到最大。谁能塞下更大的 batch,谁的吞吐就高。

## 那什么限制 batch 变大?→ 回到 KV Cache

batch 里每多一条序列,就多一份 KV cache(Part 2 算过,一条就要几百 MB)。**显存装不下更多 KV,batch 就大不了。**

于是问题精确化成:

> **在有限显存里,把 KV cache 管得越省、装得越多 → batch 越大 → 吞吐越高。**

**vLLM 的两张王牌,本质都是在回答这一个问题。** 记住这条主线,Part 4~7 全串起来了:

```
解码是内存墙(Part 3)
   └─→ 想提速就得把 batch 做大(免费午餐)
         └─→ batch 大小被 KV cache 显存卡住(Part 2)
               └─→ 谁把 KV 管得省、装得多,谁赢
                     ├─→ PagedAttention:让 KV 几乎不浪费(Part 5)
                     └─→ 连续批处理:让 batch 时刻保持满(Part 6)
```

---

<a id="part-4"></a>
# Part 4. 传统方式(HF transformers)到底浪费在哪

要讲清 vLLM 好在哪,先看它对手差在哪。HF 原生推理有**两处硬伤**,正好对应 vLLM 的两张牌。

## 硬伤一:KV 显存"预留一大块连续空间"→ 巨量浪费

HF 给每条序列的 KV cache **预留一整块连续显存,而且按"可能的最大长度"预留**。三种浪费:

| 浪费类型 | 说明 | 食堂类比 |
|---|---|---|
| **预留过度(reserved)** | 你设 `max_len=2048`,它就按 2048 占坑,哪怕这条回答只生成了 100 个 token | 每人先发一个能装 2048 的大托盘,占着 |
| **内部碎片(internal)** | 预留的 2048 里,实际用了 100,**剩下 1948 空着但被占着**,别人用不了 | 托盘里只打了二两饭,但整个托盘被你端走了 |
| **外部碎片(external)** | 显存里到处是"预留但没用满"的洞,凑不出一块新的连续大块给新请求 | 桌上散落着一堆半空托盘,却腾不出一整张干净桌子 |

论文实测:HF 这种管法,**KV 显存的有效利用率常常只有 20%~40%**。也就是说 60%~80% 的显存在空转,本来能塞下的 batch 被硬生生压小 → 吞吐上不去。

## 硬伤二:静态批处理(Static Batching)→ 队头阻塞 + GPU 空转

HF 想同时处理多条请求,得用**静态批**:凑齐一批一起 `generate`,而且**必须等这一批里最长的那条也生成完**,才能收下一批。

```
静态批(HF):一批 4 条请求,长度差异大
序列A(要生成 200 token): ████████████████████ 跑满
序列B(生成 20 token 就结束): ██░░░░░░░░░░░░░░░░░░ ← 早就答完了,却被逼着陪跑到底
序列C(生成 30 token):      ███░░░░░░░░░░░░░░░░░ ← 同上,GPU 位置白白空着
序列D(生成 200 token):     ████████████████████ 跑满
                          └────── 这段时间 B/C 的位置在空转 ──────┘
        整批必须等 A、D 跑完才能释放,新请求只能干等 → 队头阻塞
```

两个恶果:
- **队头阻塞(head-of-line blocking)**:短请求被长请求拖着,迟迟不能返回;新来的请求只能在门外排队。
- **GPU 空转**:B、C 提前结束后留下的 batch 空位没人填,算力浪费。

---

<a id="part-5"></a>
# Part 5. vLLM 王牌一:PagedAttention(把 KV 当操作系统内存管)⭐

> 对应干掉**硬伤一(KV 浪费)**。这是 vLLM 论文的招牌创新,主讲必讲。

## 灵感:抄操作系统的"虚拟内存 / 分页"

操作系统管内存时,不会给每个程序一整块连续物理内存,而是:
- 把内存切成固定大小的**页(page)**;
- 程序要多少给多少页,**物理上这些页可以东一块西一块,不用连续**;
- 用一张**页表(page table)**记录"程序看到的逻辑地址 → 实际物理页"的映射。

**PagedAttention 把这套原样搬到 KV cache 上。** 如果你没学过操作系统,换个类比也行:

> **图书馆储物柜** 📦:你存东西不用"包下一整排连续柜子",而是**要几个格子给几个格子**(哪个空给哪个,不用挨着),前台一张单子记着"你的东西在 3 号、17 号、42 号柜"。

## 具体怎么做

1. 把 KV cache 显存切成固定大小的 **KV 块(block)**,每块存**固定个数 token**的 KV(vLLM 默认一块装 16 个 token)。
2. 一条序列的 KV cache = **一串块的列表**,这些块在物理显存里**不用连续**。
3. 每条序列配一张 **块表(block table)**:逻辑上"第 0~15 个 token"→ 物理第 7 号块,"第 16~31 个"→ 物理第 3 号块……
4. 序列生成到需要新空间时,**按需分配一个新块**;序列结束,它的块**立即回收**给别人用。

```
序列A的块表:  [逻辑块0]→物理#7   [逻辑块1]→物理#3   [逻辑块2]→物理#9(刚用了2个token,还空14个)
序列B的块表:  [逻辑块0]→物理#5   [逻辑块1]→物理#1
物理显存池:  #1(B) #3(A) #5(B) #7(A) #9(A,半满) #2 #4 #6...(空闲块,谁要给谁)
```

## 好处(对着硬伤一逐条消灭)

| 传统硬伤 | PagedAttention 怎么破 |
|---|---|
| 预留过度 | **按需分块**,生成到哪要到哪,不预留最大长度 |
| 内部碎片 | 只有每条序列**最后一个块**可能没装满(最多浪费 15 个 token 的位置),浪费 < 4% |
| 外部碎片 | 块是**固定大小**的,任意空闲块都能拿来用,**不存在"凑不出连续空间"** |

论文数据:KV 显存利用率从 20~40% 拉到 **96%+**。省下的显存**全部拿去塞更大的 batch** → 吞吐暴涨。这就是 Part 3 主线的兑现。

## 彩蛋:块能"共享"(Copy-on-Write)

因为 KV 现在是"一块块"的,**多条序列可以共享同一批物理块**,直到谁要改才复制(写时复制,copy-on-write,又是抄操作系统):
- **相同前缀 prompt**:10 条请求都以同一段 system prompt 开头 → 那段的 KV **只存一份**,10 条共享。
- **并行采样(一个 prompt 生成 n 个候选答案)**:prompt 部分的 KV 共享,只有各自续写的部分才分家。
- 这进一步省显存 → batch 更大。主讲提一嘴,加分。

---

<a id="part-6"></a>
# Part 6. vLLM 王牌二:连续批处理(Continuous Batching)⭐

> 对应干掉**硬伤二(静态批的空转与阻塞)**。也叫 in-flight batching(TensorRT-LLM 的叫法)、iteration-level scheduling(按步调度)。

## 核心思想:批不是"一锅端",而是"随到随走"

静态批是"**一批人一起进、一起出**"。连续批是"**在每个解码步都重新组队**":

- 每生成完一步(每条序列各吐 1 个 token),调度器检查:谁生成了结束符(EOS)→ **立刻让它毕业、释放它的 KV 块**;
- 门口排队的新请求 → **立刻补进这个 batch 的空位**,不用等当前这批全跑完。

```
连续批(vLLM):批的成员时刻变化
步骤→   t1    t2    t3    t4    t5    t6
序列A   █     █     █     ✔走
序列B   █     ✔走
序列C   █     █     █     █     █     ✔走
新来D         →进   █     █     ✔走
新来E                     →进   █     █ ...
        └ batch 始终填满,没有空位,GPU 不空转,新请求秒进 ┘
```

## 好处

- **消灭队头阻塞**:短请求答完立刻走、立刻返回给用户,不再陪长请求跑到底。
- **GPU 不空转**:空出的 batch 位置马上被新请求填上,算力利用率拉满。
- 和 PagedAttention 是**绝配**:因为 KV 是分块的、能秒回收秒分配,"谁走谁进"才做得到——静态那种"预留一整块"根本没法这样灵活腾挪。

---

<a id="part-7"></a>
# Part 7. 合起来:为什么快 10~24 倍(一条因果链讲透)⭐

**如果你上台只能讲一页,就讲这一页。** 把前面所有点串成一条因果链:

```
① LLM 推理的瓶颈在"解码",而解码是【内存墙】问题(Part 3)
      —— GPU 算力在发呆,时间花在反复搬权重上

② 破局靠【把 batch 做大】:一次搬权重,顺带服务一整批,吞吐随 batch 涨(Part 3 免费午餐)

③ 但 batch 大小被【KV cache 显存】死死卡住(Part 2)
      —— KV 很大,显存装不下更多 KV,batch 就大不了

④ 传统 HF 又把 KV 显存浪费掉 60~80%(碎片+预留),还用静态批让 GPU 空转(Part 4)

⑤ vLLM 两张牌,精准解决 ③④:
      🃏 PagedAttention   → KV 利用率 40% 冲到 96%,省出的显存全拿去塞更大 batch(Part 5)
      🃏 连续批处理        → batch 时刻保持满、新请求秒进、GPU 不空转(Part 6)

⑥ 结果:同一张 GPU,batch 更大 + 一直满 →【吞吐翻 10~24 倍】(vLLM 论文对比 HF transformers)
```

**一句话版(背下来)**:
> "LLM 推理慢在解码,而解码是内存带宽瓶颈,解法是把 batch 做大;可 batch 被 KV cache 显存卡死。vLLM 用 PagedAttention 把 KV 显存利用率从不到一半拉到 96%,又用连续批处理让 batch 时刻满载——于是同一张卡吞吐翻十几倍。"

⚠️ **诚实提醒(主讲别吹过头)**:"10~24 倍"是**高并发、大 batch 服务**场景下、对比**原生 HF transformers** 的数字。如果你只是单条请求问一句,vLLM 的优势小得多(见 Part 12)。被问到时如实说,反而显得你真懂。

---

<a id="part-8"></a>
# Part 8. 上手:安装 + 两种用法

## 安装

vLLM 有编译好的 CUDA 轮子,直接 pip 即可(AutoDL 上先 `source /etc/network_turbo` 加速):

```bash
pip install vllm
# 拉模型走国内镜像:
export HF_ENDPOINT=https://hf-mirror.com
```

- vLLM 对 **PyTorch / CUDA 版本敏感**:它会**指定并可能重装一个特定版本的 torch**。⚠️ 这和你 VAR 那边"别动 torch"的原则冲突——**所以强烈建议给 vLLM 单独开一个 conda / venv 环境**,别和 VAR 的环境混用。
- 首次 `pip install vllm` 会拉不少依赖,耐心等。

## 用法 A:离线批量推理(`LLM` 类)—— 你测速、跑批用这个

一次性喂一堆 prompt,拿回全部结果。**这就是你做师兄任务①(测速)要用的**:

```python
from vllm import LLM, SamplingParams

# 1) 加载模型(这步耗时,包含权重加载 + 显存 profiling,不计入测速)
llm = LLM(
    model="Qwen/Qwen3-3B",       # 模型名,自动从 HF(镜像)下载
    dtype="bfloat16",            # 精度
    gpu_memory_utilization=0.9,  # 允许 vLLM 吃 90% 显存(权重 + KV池)
    # tensor_parallel_size=2,    # 多卡才加,见 Part 10
)

# 2) 采样参数(相当于 HF 的 generate 参数)
sp = SamplingParams(temperature=0.0, max_tokens=256)  # temp=0 → 贪心,和 HF do_sample=False 对齐

# 3) 一次性喂一批 prompt,vLLM 内部自动连续批 + PagedAttention
prompts = ["你好,介绍一下你自己", "用一句话解释什么是注意力机制", "写一首关于秋天的诗"]
outputs = llm.generate(prompts, sp)

# 4) 取结果
for o in outputs:
    print(o.prompt, "→", o.outputs[0].text)
```

> 想让 vLLM 自动套聊天模板(chat template)?用 `llm.chat(messages, sp)`,传的是 `[{"role":"user","content":...}]` 列表,它内部帮你 `apply_chat_template`。测速时为了和 HF 完全对齐,我更推荐**自己套好模板再传 `generate`**(Part 11 就这么做)。

## 用法 B:起一个 OpenAI 兼容的 API 服务(部署用)

一行命令把模型变成一个**和 OpenAI 接口一模一样**的本地服务,别的程序(甚至 ReAct 的 agent 框架)可以直接当 OpenAI 用:

```bash
vllm serve Qwen/Qwen3-3B --dtype bfloat16 --gpu-memory-utilization 0.9 --port 8000
# 多卡: 加 --tensor-parallel-size 2
```

然后任何地方都能这样调(注意 `base_url` 指向你本地):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")  # 本地服务 key 随便填
resp = client.chat.completions.create(
    model="Qwen/Qwen3-3B",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

> 这个"本地 OpenAI 接口"非常重要:**你后面做 ReAct / Tool Use / MCP RL,agent 框架基本都只认 OpenAI 接口**。用 vLLM serve 起一个,就能白嫖 GPT 那套代码接自己的开源模型。主讲时点出这个"承上启下"的作用,能把 vLLM 和 ReAct 两块任务串成一条线。

---

<a id="part-9"></a>
# Part 9. 关键参数(调不好就 OOM / 变慢)

启动 `LLM(...)` 或 `vllm serve` 时,这几个参数最常调,面试/主讲也常问:

| 参数(Python / 命令行) | 作用 | 怎么设 |
|---|---|---|
| `gpu_memory_utilization` / `--gpu-memory-utilization` | vLLM 允许占用的显存比例(权重 + KV 池) | 默认 0.9。**和别的进程共享卡就调低**(如 0.5);想塞更大 batch 就调高(0.95,风险是易 OOM) |
| `max_model_len` / `--max-model-len` | 单条序列最大长度(prompt+生成) | **不设会按模型默认(可能很大)**,导致 KV 池预算爆炸、启动报显存不足。**手动设小(如 2048、4096)是最常见的 OOM 解法** |
| `tensor_parallel_size` / `--tensor-parallel-size` | 用几张卡做张量并行(Part 10) | 单卡=1(默认);双卡=2。必须能整除注意力头数 |
| `max_num_seqs` / `--max-num-seqs` | 一个 batch 最多同时跑多少条序列 | 默认 256。显存紧就调小 |
| `dtype` / `--dtype` | 精度 | `bfloat16`(推荐)/ `float16` / `auto` |
| `enforce_eager` / `--enforce-eager` | 关掉 CUDA Graph 优化 | 默认关(即默认用 CUDA Graph 加速)。**调试/显存实在不够**时加上它省点显存,但会慢一点 |
| `quantization` / `--quantization` | 量化(awq/gptq/fp8 等) | 想用量化模型省显存时设;入门可先不碰 |

> **90% 的 vLLM 启动报错是 "CUDA out of memory"**,标准三连解法:①调低 `gpu_memory_utilization`;②调小 `max_model_len`;③调小 `max_num_seqs`。记这三个,现场演示崩了也能救。

---

<a id="part-10"></a>
# Part 10. 多卡:张量并行(和你会的 DDP 数据并行区别)

> 师兄任务里 SWIFT 那条要"多卡配置"。vLLM 这边多卡也顺带讲清,因为概念相通,而且**这里能秀你已经会的 DDP**。

## 关键区分:vLLM 多卡用的是"模型并行",不是你学的"数据并行"

你在 DDP / Accelerate 里学的是**数据并行(data parallel)**:每张卡放**一整份完整模型**,把**数据**切开分给各卡。那是**训练**的玩法。

vLLM 推理默认用 **张量并行(tensor parallel,TP)**,属于**模型并行(model parallel)**:把**同一个模型的权重横切开**,一层的计算摊到多张卡上,每卡只存**一部分**权重,算完用 all-reduce 合并。

| | 数据并行(你会的 DDP) | 张量并行(vLLM 推理) |
|---|---|---|
| 每张卡放什么 | **完整**模型一份 | 模型的**一部分**权重 |
| 切的是什么 | 切**数据** | 切**模型(权重矩阵)** |
| 解决什么 | 训练提速(数据太多) | ① 单卡装不下大模型 ② 推理再提速 |
| 你在哪见过 | `step01~04`、`train_sft.py` | 就是现在,`tensor_parallel_size=2` |

> 主讲金句:"DDP 是**人手一份卷子分开做**(数据并行);vLLM 张量并行是**一张卷子几个人分工、一人做几道题再拼答案**(模型并行)。"—— 一句话区分两个最容易混的概念,老师会记住你。

## 怎么用

就一个参数(卡数要能整除模型的注意力头数,一般 2/4/8 都行):

```python
llm = LLM(model="Qwen/Qwen3-3B", tensor_parallel_size=2)   # 双卡
```
```bash
vllm serve Qwen/Qwen3-3B --tensor-parallel-size 2          # 双卡服务
```

> 小模型(3B)其实**单卡就够跑**,用双卡 TP 主要是"体验 + 装得下更大 batch/更长上下文"。真需要 TP 的是 32B、70B 这种单卡装不下的。任务要"多卡体验",你演示 `tensor_parallel_size=2` 能跑通即可。

---

<a id="part-11"></a>
# Part 11. 师兄任务①实战:HF vs vLLM 测速(Qwen3-3B)⭐

> 任务原文:**"对比 Huggingface 推理和 VLLM 推理速度的差异(QWen3-3B),理解原因。"**
> "理解原因" = Part 3+7 那条因果链。"对比速度" = 下面这两个脚本。建议存成 `week3/bench_hf.py` 和 `week3/bench_vllm.py`,或合成一个,分开跑(**因为 HF 和 vLLM 常要不同环境**,别在同一进程里混跑)。

## 测速要公平:三条铁律

1. **同样的输入**:同一批 prompt、同样的 `max_tokens`(如都 256)、同样贪心解码(HF `do_sample=False` ↔ vLLM `temperature=0`)。
2. **不计模型加载时间**:计时器**只包住生成那段**,`from_pretrained` / `LLM(...)` 放计时器外面。
3. **先热身(warmup)**:第一次跑有 CUDA 初始化 / 图捕获开销,先空跑一次不计时,再正式测。
4. **测吞吐,不只是测单条**:vLLM 的主场是**一批请求**。你要造**几十上百条 prompt 一起测**,才看得出差距;只测 1 条会严重低估 vLLM。

## 指标:算什么

- **吞吐(throughput)= 总生成 token 数 / 总耗时(tok/s)** ← 最能体现 vLLM 优势,主看这个。
- (可选)单条平均延迟、首 token 延迟。

## 脚本 A:HuggingFace 基线(`bench_hf.py`)

用**批量(batched)** generate,这已经是 HF 能做到的较公平基线(比 for 循环一条条问快;但仍是静态批 + 无 PagedAttention):

```python
# bench_hf.py  —— HuggingFace transformers 推理基线
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-3B"
N = 64          # 请求条数(造 64 条并发,才测得出差距)
MAX_NEW = 256   # 每条生成多少 token

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"                       # ★批量生成必须左填充,否则结果错
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

# 造 N 条 prompt(套上 chat 模板,和 vLLM 那边保持一致)
questions = [f"用三句话介绍第 {i} 个有趣的科学事实" for i in range(N)]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True) for q in questions]
enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")

@torch.no_grad()
def run():
    return model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                          pad_token_id=tok.pad_token_id)

# 热身(不计时)
_ = run(); torch.cuda.synchronize()

# 正式测
t0 = time.time()
out = run(); torch.cuda.synchronize()
dt = time.time() - t0

gen_tokens = (out.shape[1] - enc["input_ids"].shape[1]) * N   # 每条新增 token × 条数
print(f"[HF]   {N} 条 × {MAX_NEW} token | 耗时 {dt:.1f}s | 吞吐 {gen_tokens/dt:.0f} tok/s")
```

## 脚本 B:vLLM(`bench_vllm.py`)

```python
# bench_vllm.py  —— vLLM 推理
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-3B"
N = 64
MAX_NEW = 256

tok = AutoTokenizer.from_pretrained(MODEL)
questions = [f"用三句话介绍第 {i} 个有趣的科学事实" for i in range(N)]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True) for q in questions]

llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)

# 热身(不计时)
_ = llm.generate(prompts[:2], sp)

# 正式测
t0 = time.time()
outs = llm.generate(prompts, sp)
dt = time.time() - t0

gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"[vLLM] {N} 条 × {MAX_NEW} token | 耗时 {dt:.1f}s | 吞吐 {gen_tokens/dt:.0f} tok/s")
```

## 跑法 + 怎么解读

```bash
# 两个环境分别跑(vLLM 可能重装 torch,别和 HF 混一个 env)
python bench_hf.py
python bench_vllm.py
```

**预期(具体数字看你的卡,方向一定是 vLLM 大胜)**:

| 场景 | HF(batched) | vLLM | 说明 |
|---|---|---|---|
| N=1(单条) | 基准 | 略快或相近 | ⚠️ 单条看不出差距,别只测这个 |
| N=64(并发) | 慢 | **快数倍~十几倍** | 差距在这里爆发 |
| 把 N 加到 128/256 | 可能 OOM 或线性变慢 | 吞吐继续涨(连续批 + PagedAttention 撑得住) | vLLM 的护城河 |

**解读词(写进你的对比报告 / 讲稿)**:
> "HF 批量生成用**静态批**,必须等最长序列生成完才释放,且 KV 显存**预留浪费严重**,并发一大就变慢甚至 OOM;vLLM 用 **PagedAttention** 把 KV 显存利用率拉满、塞下更大 batch,再用**连续批处理**让 GPU 不空转——所以**并发越高,vLLM 领先越多**。这印证了'解码是内存墙、提速靠做大 batch'这条原理。"

> 💡 想让对比更有说服力:**扫一组 N(1、8、32、64、128)画一条"吞吐 vs 并发数"曲线**——HF 很快见顶甚至掉头,vLLM 一路往上。这张图放 slide 里,比一句"快 10 倍"有力得多。

---

<a id="part-12"></a>
# Part 12. 什么时候 vLLM 帮助没那么大 + 坑速查

## vLLM 不是万能(主讲被问到时的加分回答)

| 场景 | vLLM 还香吗 | 为什么 |
|---|---|---|
| 高并发在线服务、批量离线推理 | ⭐ 主场,香爆 | 连续批 + PagedAttention 全用上 |
| **只有单条请求、且要极低延迟** | 一般 | 没 batch 可拼,PagedAttention 省的显存也用不上;此时 TensorRT-LLM 这类更极致 |
| **超短生成(如只生成 1~5 个 token 的分类)** | 一般 | 解码步太少,优化空间小 |
| 显存极度紧张、只想跑个 demo | 未必 | vLLM 会预吃一大块显存建 KV 池,小打小闹时反而"重" |

## 坑速查

| 现象 | 原因 | 解决 |
|---|---|---|
| `CUDA out of memory`(启动就崩) | KV 池 / 模型太大 | 三连:调低 `gpu_memory_utilization`、调小 `max_model_len`、调小 `max_num_seqs` |
| 装完 vLLM,VAR / 别的项目跑不了了 | vLLM 重装了特定版 torch,冲突 | **给 vLLM 单开一个 env**,别和 VAR 混 |
| 模型下载失败 / 卡住 | 直连 huggingface.co | `export HF_ENDPOINT=https://hf-mirror.com`(+ AutoDL 的 `source /etc/network_turbo`) |
| `Qwen/Qwen3-3B` 拉不到(404 / 名字不对) | 该 size 名字对不上或未发布 | 去 hf-mirror 确认实际有的名字,换 `Qwen3-4B` 或 `Qwen2.5-3B`,**不影响对比方法论** |
| 多卡 `tensor_parallel_size=2` 报错 | 卡数不整除注意力头数 / 卡间通信没配好 | 换能整除的卡数;确认两张卡可见(`CUDA_VISIBLE_DEVICES`) |
| 测出来 vLLM 没比 HF 快多少 | 只测了 1~2 条请求 | **把并发 N 加大**(64+),vLLM 的优势要在并发下才显现 |
| 结果乱码 / 明显不对(HF 那边) | 批量生成忘了 `padding_side="left"` | HF 批量 generate 必须左填充 |

---

<a id="part-13"></a>
# Part 13. 和别的推理引擎的关系(一句话各是什么)

主讲可能被问"那 XX 和 vLLM 比呢",备好这张表:

| 引擎 | 谁做的 | 一句话定位 | 和 vLLM 关系 |
|---|---|---|---|
| **HF transformers** | HuggingFace | 通用、易用,但推理**没为吞吐优化** | vLLM 的**对比基线**(就是你任务①要打败的) |
| **vLLM** | UC Berkeley | **PagedAttention + 连续批**,开源部署事实标准 | 本文主角 |
| **TGI**(Text Generation Inference) | HuggingFace | HF 官方的生产级推理服务,也用了连续批 | 同类竞品,vLLM 论文里也和它比过 |
| **TensorRT-LLM** | NVIDIA | 榨干 N 卡的极致性能,但要编译、门槛高 | 更快但更难用;追求极致低延迟时用 |
| **SGLang** | 也源自 Berkeley 一系 | 强在**复杂控制流 / 结构化输出 / 前缀共享**(RadixAttention) | vLLM 的"近亲兼对手",做 agent / 多轮很能打 |

一句话记忆:**"vLLM = 开源部署的默认选项;要极致性能上 TensorRT-LLM;要玩复杂 agent 流程看 SGLang。"**

---

<a id="part-14"></a>
# Part 14. 自测(上台前自己过一遍)

答得上,你就是真懂,不是背 API:

1. LLM 推理分哪两个阶段?各自是 compute-bound 还是 memory-bound?(Part 1、3)
2. **为什么解码阶段"把 batch 做大"几乎是免费的午餐?**(Part 3)⭐核心
3. 是什么东西限制了 batch 不能无限大?(Part 2、3)
4. KV cache 是干嘛的?不缓存会怎样?它为什么吃显存?(Part 2)
5. 传统 HF 推理的**两处硬伤**是什么?(Part 4)
6. **PagedAttention 抄了操作系统的什么机制?它怎么把 KV 利用率从 40% 拉到 96%?**(Part 5)⭐
7. 连续批处理解决了静态批的什么问题?为什么它和 PagedAttention 是绝配?(Part 6)
8. **用一条因果链说明"vLLM 为什么比 HF 快 10 倍"**(Part 7)⭐必考
9. vLLM 多卡用的张量并行,和你学的 DDP 数据并行有什么本质区别?(Part 10)
10. 什么场景下 vLLM 优势不明显?(Part 12)

> 第 2、6、8 是灵魂三问。这三个答得流畅,主讲稳了。

---

<a id="part-15"></a>
# Part 15. 术语表(最后回填)

| 术语 | 一句话 |
|---|---|
| **吞吐 / throughput** | 单位时间生成的总 token 数(tok/s)。vLLM 优化的核心指标 |
| **延迟 / latency** | 单条请求从发出到拿到结果的时间(或首 token 时间) |
| **预填充 / prefill** | 推理第一阶段:把整个 prompt 并行算一遍,compute-bound |
| **解码 / decode** | 推理第二阶段:一个 token 一个 token 串行生成,memory-bound |
| **KV Cache** | 缓存过去 token 的 K、V 向量,避免重算;吃显存的大头 |
| **内存墙 / memory-bound** | 瓶颈是显存带宽(搬权重)而非算力;解码就是这种 |
| **算力密集 / compute-bound** | 瓶颈是计算量;预填充是这种 |
| **batch / 批** | 一次同时处理的序列数。越大吞吐越高(受 KV 显存限制) |
| **PagedAttention** | vLLM 招牌:把 KV cache 分成固定块、像 OS 分页一样管,近零浪费 |
| **KV 块 / block** | KV cache 的固定大小单元(vLLM 默认 16 token/块) |
| **块表 / block table** | 记录"逻辑块 → 物理块"映射,类比 OS 页表 |
| **连续批处理 / continuous batching** | 每个解码步重组 batch,谁完成谁走、新请求秒进;消除空转 |
| **静态批 / static batching** | 传统做法:一批一起进出,必须等最长序列跑完 |
| **队头阻塞 / head-of-line blocking** | 短请求被前面的长请求拖住,迟迟不返回 |
| **张量并行 / TP** | 模型并行的一种:把权重横切到多卡;vLLM 多卡用它 |
| **数据并行 / DP** | 每卡一份完整模型、切数据;你 DDP 学的那种(训练用) |
| **GQA** | 多个 Query 头共享一组 KV 头,砍 KV cache;Qwen3 等在用 |
| **copy-on-write** | 多序列共享 KV 块,要改时才复制;省显存(如共享 prompt 前缀) |
| **gpu_memory_utilization** | vLLM 允许占的显存比例,默认 0.9 |
| **max_model_len** | 单序列最大长度;OOM 时优先调小它 |

---

## 📌 主讲三句话锦囊(全场记不住别的,记这三句)

1. **原理**:"LLM 推理慢在解码,解码是内存墙,提速靠把 batch 做大,而 batch 被 KV cache 显存卡死。"
2. **vLLM 两张牌**:"PagedAttention 把 KV 显存当操作系统内存分页管、利用率从 40% 冲到 96%;连续批处理让 batch 时刻满载、GPU 不空转。"
3. **结果**:"同一张卡,batch 更大又一直满,吞吐翻十几倍——并发越高领先越多。"

> 下一份写 **ReAct 论文精读**(你分工的另一半)。ReAct 正好接得上 vLLM:vLLM serve 起个 OpenAI 接口,ReAct 的 agent 就用它当大脑,再配 Tool Use RL——这条线你一个人就能讲通,主讲会很有底气。
