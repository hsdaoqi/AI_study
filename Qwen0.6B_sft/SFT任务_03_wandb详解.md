# wandb 详解(SFT任务第③项 · 实验记录与可视化)

---

## 目录
- [Part A. 是什么 & 为什么](#part-a-是什么--为什么)
- [Part B. 核心概念(4 个)](#part-b-核心概念4-个)
- [Part C. 基础 API(三步 + 细节)](#part-c-基础-api三步--细节)
- [Part D. 师兄的两个重点:tag 分组 + 离线上传](#part-d-师兄的两个重点tag-分组--离线上传)
- [Part E. 登录 & 环境变量](#part-e-登录--环境变量)
- [Part F. 多卡怎么用(rank0 + Accelerate 原生)](#part-f-多卡怎么用rank0--accelerate-原生)
- [Part G. 完整示例(拼起来)](#part-g-完整示例拼起来)
- [Part H. 常见坑](#part-h-常见坑)
- [Part I. 自测](#part-i-自测)

---

# Part A. 是什么 & 为什么

## A0. 一句话

> **wandb(Weights & Biases,读"万 di bi")= 训练过程的"行车记录仪 + 云端仪表盘"。**
> 训练时把 loss、学习率、显存这些数,一行代码发给它,它自动画成**实时更新的曲线图**,存在网页上,随时随地打开看。

## A1. 它解决什么痛点(对比土办法)

| 痛点(只用 print) | wandb 怎么解决 |
|---|---|
| loss 只能 print 刷屏,训完就没了 | **永久存云端**,历史随便翻 |
| 想看曲线得手动复制到 Excel 画 | **自动实时画曲线**,浏览器刷新就更新 |
| 跑了 5 组不同超参,分不清谁是谁 | 每次训练 = 一个 **run**,自动记下用了什么参数 |
| 训练在 AutoDL、人在宿舍 | 云端仪表盘,**手机都能看**训练到哪了 |
| 想对比"改学习率前后" | 多个 run **叠一张图**上对比 |

## A2. 它在你这个任务里的角色

> **关于"权重曲线"这个说法**:任务里写的是"上传权重曲线",有两种理解——
> 1. **训练/loss 曲线**(最常见,指 loss 随 step 下降的曲线)→ 用 `wandb.log`(Part C2)。
> 2. **权重/梯度的分布曲线**(模型参数随训练怎么变)→ 用 `wandb.watch`(Part C5)。
>
> 我们脚本里**两个都配上**:loss 曲线是必须的,权重/梯度直方图用 `wandb.watch` 一行就能加。

---

# Part B. 核心概念(4 个)

记住这个层级,整个 wandb 就通了:

```
Project(项目)  ── 一个任务,底下装很多次训练
   └── Run(一次训练)  ── 跑一次 = 一个 run,有唯一名字
         ├── config    ── 这次用的超参(lr、batch_size、model...)
         ├── metrics   ── 训练中记录的数(loss、acc...)→ 自动画成曲线
         └── artifacts ── 存的文件(模型权重、数据集...)→ 带版本管理
```

| 概念 | 是什么 | 类比 |
|---|---|---|
| **project** | 一个课题,底下很多次训练 | 一个文件夹 |
| **run** | **跑一次训练 = 一个 run** | 文件夹里的一本记录本 |
| **config** | 这次 run 的超参数 | 记录本封面写的"实验条件" |
| **metric** | 训练中记的数,自动画曲线 | 记录本里的数据表 → 折线图 |
| **artifact** | 存的文件 + 版本 | 记录本夹的附件 |

> 一次训练脚本从头跑到尾 = **一个 run**。改了超参再跑一次 = **另一个 run**。同一个 project 下的多个 run 可以在网页上叠加对比。

---

# Part C. 基础 API(三步 + 细节)

**核心就三个函数:`init` 开局 → `log` 记数 → `finish` 收尾。** 下面逐个讲透,再补几个进阶的。

## C1. `wandb.init()` —— 开一个 run

```python
import wandb

run = wandb.init(
    project="qwen3-sft",         # 【必填】项目名(不存在会自动建)
    name="run-lr2e-5",           # 这次 run 的显示名(不填会随机起,如 "fanciful-dawn-3")
    config={                     # 【强烈建议】记下超参,之后能按它筛选/对比
        "lr": 2e-5,
        "batch_size": 4,
        "epochs": 3,
        "model": "Qwen3-0.6B",
    },
    tags=["sft", "qwen3-0.6b"],  # 给这个 run 打标签(用于筛选,见 Part D)
    notes="第一次跑通 SFT",       # 备注(可选)
    # mode="offline",            # 在线/离线,一般用环境变量控制(见 Part D2)
)
```

- 通常在**训练脚本最开头**调一次。
- 返回一个 `run` 对象,后面可以用 `run.log(...)`(和 `wandb.log(...)` 等价)。
- `config` 存进去后,网页上每个 run 都能看到它用了什么超参——**这是区分不同实验的关键,一定要填**。

## C2. `wandb.log()` —— 记录指标(核心中的核心)⭐

```python
wandb.log({"train/loss": 0.53, "train/lr": 2e-5})
```

**规则:**
- 传一个字典:**键 = 图的名字,值 = 这一步的数值**。
- 同一个键**反复 log**,wandb 自动把这些点连成**一条曲线**。
- 横轴默认是"log 的次数"(internal step),也可以自己指定(见下)。

**键名里的 `/` 会自动分组(重要!):**
```python
wandb.log({"train/loss": ...})   # 进 "train" 分区
wandb.log({"train/lr":   ...})
wandb.log({"eval/loss":  ...})   # 进 "eval" 分区
wandb.log({"eval/acc":   ...})
```
→ 仪表盘上自动出现 **train** 和 **eval** 两个面板组,图各归各。**这就是控制"图怎么分组"最直接的方式**(Part D 细讲)。

**关于横轴 step:**
```python
# 方式1:让 wandb 自己数(每调一次 log,step +1)
wandb.log({"train/loss": loss})

# 方式2:自己指定 step(推荐,多个指标对齐同一横轴)
wandb.log({"train/loss": loss}, step=global_step)
```
> ⚠️ 坑:如果你在同一个 step 里分开 log 好几次(比如先 log train 再 log eval),不指定 step 会导致 step 错乱/告警。**最佳实践:每个训练 step 用一次 `log`,或统一传 `step=global_step`。**(见 Part H)

## C3. `wandb.finish()` —— 收尾

```python
wandb.finish()
```
- 训练结束调一次,把缓冲区数据刷完、正常关闭 run。
- 脚本正常结束一般会自动调,但**显式写上更稳**(尤其多卡/异常时)。

## C4. `wandb.config` —— 超参数

除了在 `init` 里传,也可以后补:
```python
wandb.config.update({"warmup_steps": 100})
wandb.config.lr        # 读取
```
> 好处:网页上可以按 config 排序、筛选。比如"给我看所有 lr=2e-5 的 run"。

## C5. `wandb.watch()` —— 自动记录权重/梯度曲线(对应"权重曲线")

```python
wandb.watch(model, log="all", log_freq=100)
# log="gradients"(默认)/ "parameters" / "all" / None
# log_freq=100:每 100 个 batch 记一次
```
- 它会自动把模型的**权重分布**和**梯度分布**记成**直方图**,随训练更新。
- 如果师兄说的"权重曲线"指的是这个(参数/梯度随训练怎么变),就靠它。
- **一行就能加**,放在 `init` 之后、训练循环之前。
> 多卡时只在主进程 watch(见 Part F)。

## C6. `wandb.Artifact` —— 上传模型权重(带版本)

如果"上传权重"指的是**把训练好的模型文件传上去存档**:
```python
# 训练完,保存模型后:
model.save_pretrained("output/sft")           # 先存成 HF 格式
artifact = wandb.Artifact("qwen3-sft-model", type="model")
artifact.add_dir("output/sft")                # 把整个目录加进去
wandb.log_artifact(artifact)                  # 上传(带版本号 v0, v1...)
```
- artifact 带**版本管理**:每次上传自动 v0→v1,能回溯。
- ⚠️ 模型文件大(1.4GB),离线/网差时慎传;通常**先本地存,联网了再传**。

## C7. 其他能 log 的东西(了解)

`wandb.log` 不止能记数字,还能记:
```python
wandb.log({"样例图": wandb.Image(img)})            # 图片
wandb.log({"分布": wandb.Histogram(tensor)})       # 直方图
wandb.log({"预测表": wandb.Table(data=..., columns=...)})  # 表格
```
> SFT 任务主要用数字曲线(loss/lr),这些先知道即可。

---

# Part D. 两个重点:tag 分组 + 离线上传

## D1. 「为不同的图打 tag」—— 三种分组方式,别搞混

wandb 里"分组"有三层,"不同图打 tag"**最可能是第 1 种**,但三种都讲清:

### ① key 前缀 `/` —— 给**图表**分区(最常用,优先用这个)⭐
```python
wandb.log({
    "train/loss": ...,   # ┐
    "train/lr":   ...,   # ┴→ 仪表盘 "train" 面板组
    "eval/loss":  ...,   # ┐
    "eval/acc":   ...,   # ┴→ 仪表盘 "eval" 面板组
})
```
→ **控制"图在仪表盘上怎么归类"**。写脚本时我会统一用 `train/xxx`、`eval/xxx` 前缀,让 loss、lr、eval 各成一块,不糊成一团。**这基本就是师兄要的"不同图分门别类"。**

### ② `tags` —— 给**整个 run** 贴标签(用于筛选 run)
```python
wandb.init(project="qwen3-sft", tags=["sft", "qwen3-0.6b", "2gpu", "baseline"])
```
→ 网页上可以按 tag 筛:"给我看所有 `2gpu` 的 run""所有 `baseline` 的 run"。**不是控制图,是控制 run 的分类。**

### ③ `group` —— 把多个 run 归成一组(多卡/多次实验)
```python
wandb.init(project="qwen3-sft", group="exp-lr-scan", job_type="train")
```
→ 常用于:同一个实验开了多个 run(比如多机多卡每个进程一个 run),用 group 归拢显示。

> **小结**:要让**图**分类 → 用 `/` 前缀(①);要让**run**能筛 → 用 `tags`(②)。**脚本里我①②都配**,这样"打 tag"这条要求怎么理解都覆盖到。

## D2. 「离线保存后上传」—— AutoDL 连不上外网的退路 ⭐

wandb 是**云端**服务(服务器在国外),**AutoDL 经常连不上**。所以要么在线实时传,要么离线存本地、之后再同步。靠环境变量 `WANDB_MODE` 切换:

| `WANDB_MODE` | 行为 | 什么时候用 |
|---|---|---|
| `online`(默认) | 实时传云端 | 能连外网 |
| `offline` | 只存本地磁盘,不联网 | **AutoDL 连不上时** |
| `disabled` | 完全关闭(啥也不记) | debug 脚本时 |

### 完整离线工作流

**Step 1:训练时设离线模式**
```bash
export WANDB_MODE=offline      # 训练脚本跑之前设这个
python train.py                # 或 accelerate launch train.py
```
→ 数据存到本地 `./wandb/offline-run-<时间>-<id>/` 文件夹,**全程不联网,不会因为连不上而报错/卡住**。

**Step 2:训练完,把离线记录同步到云端**
```bash
# 有网的环境下(比如 AutoDL 能连的时段,或把 wandb/ 目录下载到本地):
wandb login                                    # 先登录(见 Part E)
wandb sync ./wandb/offline-run-20260707_xxxxx  # 同步某一个离线 run
# 或者一键同步所有离线 run:
wandb sync --sync-all
```
→ 同步完,网页上就能看到完整曲线,和在线跑的一模一样。

> **实战策略(脚本会这么写)**:
> - **优先在线**:先试 `WANDB_MODE=online`,能连就实时传(师兄说的"优先动态上传")。
> - **连不上就离线**:AutoDL 连不上,就 `export WANDB_MODE=offline` 跑,训完 `wandb sync` 补传。
> - 稳妥起见,可以**直接离线跑**(绝不会因为网卡住训练),训完统一 sync。二选一,我在脚本里都给你留好开关。

> 📁 提示:离线文件默认在当前目录 `./wandb/`。可以用 `export WANDB_DIR=/root/autodl-tmp/wandb` 把它放到数据盘(AutoDL 系统盘小),别塞满。

---

# Part E. 登录 & 环境变量

## 注册 + 拿 key

1. 去 **wandb.ai** 注册(免费)。
2. 打开 **wandb.ai/authorize**,复制你的 **API key**(一串字符)。

## 登录(三选一)

```bash
# 方式1:交互式,粘贴 key(只需一次,之后记住在 ~/.netrc)
wandb login

# 方式2:直接带 key
wandb login <你的API_KEY>

# 方式3:用环境变量(适合脚本/服务器,不想交互)
export WANDB_API_KEY=<你的API_KEY>
```
> **离线模式(`WANDB_MODE=offline`)不需要登录**也能跑;但**在线上传 / `wandb sync` 时需要**登录。

## 常用环境变量总表

| 环境变量 | 作用 | 例子 |
|---|---|---|
| `WANDB_MODE` | 在线/离线/关闭 | `offline` |
| `WANDB_API_KEY` | 登录 key | `xxxxx` |
| `WANDB_DIR` | 本地文件存哪 | `/root/autodl-tmp/wandb` |
| `WANDB_PROJECT` | 默认项目名 | `qwen3-sft` |
| `WANDB_NAME` | 默认 run 名 | `run1` |
| `WANDB_SILENT` | 静默日志 | `true` |

> 好处:这些用环境变量设,脚本里就不用写死,换环境只改变量。

---

# Part F. 多卡怎么用(rank0 + Accelerate 原生)

## F1. 为什么只能主进程(rank0)记

多卡训练时,**每张卡是一个独立进程**(你 DDP 笔记里学的)。如果每个进程都 `wandb.init` + `wandb.log`:
- 会创建 **N 个重复的 run**,曲线各传一份,乱套。
- 所以:**只让主进程(rank0 / main process)碰 wandb**,其他进程跳过。

这和你 DDP 学的 `if rank == 0: print(...)` 是**同一个道理**。

## F2. 写法一:原生 wandb + `is_main_process`(直白)

```python
from accelerate import Accelerator
accelerator = Accelerator()

# 只在主进程 init
if accelerator.is_main_process:
    wandb.init(project="qwen3-sft", name="run1", config={...}, tags=["sft"])

for step, batch in enumerate(loader):
    loss = train_step(batch)
    # 只在主进程 log
    if accelerator.is_main_process:
        wandb.log({"train/loss": loss.item()}, step=step)

if accelerator.is_main_process:
    wandb.finish()
```
- 简单直白,但每次 log 都要包 `if`,略啰嗦。

## F3. 写法二:Accelerate 原生 tracker(推荐,自动只在主进程)⭐

Accelerate **内置了 wandb 集成**,帮你自动处理"只在主进程记",代码更干净:

```python
from accelerate import Accelerator

# ① 告诉 Accelerator 用 wandb
accelerator = Accelerator(log_with="wandb")

# ② 初始化 tracker(自动只在主进程建 run)
accelerator.init_trackers(
    project_name="qwen3-sft",
    config={"lr": 2e-5, "batch_size": 4, "epochs": 3},   # 超参
    init_kwargs={"wandb": {                               # 传给 wandb.init 的额外参数
        "name": "qwen3-0.6b-sft",
        "tags": ["sft", "qwen3-0.6b", "2gpu"],
    }},
)

for step, batch in enumerate(loader):
    loss = train_step(batch)
    # ③ 用 accelerator.log,自动只在主进程记
    accelerator.log({"train/loss": loss.item(), "train/lr": cur_lr}, step=step)

# ④ 收尾
accelerator.end_training()
```

- `Accelerator(log_with="wandb")`:声明用 wandb 当 tracker。
- `accelerator.init_trackers(...)`:自动**只在主进程**建 run。
- `accelerator.log(...)`:自动**只在主进程**记,不用自己写 `if`。
- `accelerator.end_training()`:收尾(相当于 `wandb.finish`)。
- 需要拿底层 wandb run 时:`accelerator.get_tracker("wandb").run`(比如给 `wandb.watch` 用)。

> **我们的 SFT 脚本会用写法二**(F3),因为任务就是用 Accelerate,原生集成最省事、最不容易错。离线还是照样用 `export WANDB_MODE=offline` 控制。

---

# Part G. 完整示例(拼起来)

一个"能直接改用"的骨架,把 Part C/D/F 串起来(Accelerate 版):

```python
import os
from accelerate import Accelerator

# 离线开关:AutoDL 连不上就设 export WANDB_MODE=offline(见 Part D2)
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
    project_name="qwen3-sft",
    config={"lr": 2e-5, "batch_size": 4, "epochs": 3, "model": "Qwen3-0.6B"},
    init_kwargs={"wandb": {"name": "qwen3-0.6b-sft", "tags": ["sft", "qwen3", "accelerate"]}},
)

global_step = 0
for epoch in range(EPOCHS):
    for batch in loader:
        loss = train_step(batch)
        # 用 / 前缀分组图表;只在主进程记(accelerator.log 自动处理)
        accelerator.log(
            {"train/loss": loss.item(), "train/lr": get_lr(), "epoch": epoch},
            step=global_step,
        )
        global_step += 1

accelerator.end_training()
```

配合运行:
```bash
# 在线(能连外网):
accelerate launch train.py

# 离线(AutoDL 连不上):
export WANDB_MODE=offline
export WANDB_DIR=/root/autodl-tmp/wandb    # 存数据盘,别塞满系统盘
accelerate launch train.py
# 训完,有网时:
wandb sync /root/autodl-tmp/wandb/offline-run-*
```

---

# Part H. 常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| 卡在 `wandb: Currently logged in as...` / 联网超时 | AutoDL 连不上 wandb 云 | `export WANDB_MODE=offline`,训完 `wandb sync` |
| 开了 N 个重复 run | 每个进程都 init 了 | 只在主进程 init(F1);或用 `accelerator`(F3) |
| step 错乱 / `step must be monotonically increasing` 告警 | 同一步分开 log 多次、没给 step | 每步一次 log,或统一 `step=global_step` |
| 曲线是锯齿/太密 | 每个 micro-batch 都 log | 每 N 步 log 一次,或 log 平均值 |
| 系统盘满了 | 离线文件堆在 `./wandb/` | `export WANDB_DIR=/root/autodl-tmp/wandb` |
| `wandb sync` 传不上 | 没登录 | 先 `wandb login` |
| 只想调试不想记 | —— | `export WANDB_MODE=disabled` |

---

# Part I. 自测

1. wandb 的三个核心函数是哪三个,各干嘛?(Part C)
2. `wandb.log({"train/loss": x})` 里那个 `/` 有什么用?和 `tags` 有什么区别?(Part D1)
3. AutoDL 连不上外网,怎么用 wandb?训练时设什么、训完用什么命令补传?(Part D2)⭐
4. 多卡训练时,为什么只能让主进程记 wandb?Accelerate 里怎么优雅解决?(Part F)
5. 任务说"上传权重曲线",可能是哪两种?分别用什么 API?(Part A2 / C2 / C5)
6. `wandb.watch` 和 `wandb.log` 分别记什么?(Part C)

> 第 3、4 题是这个任务的关键(离线 + 多卡)。答得上,你就能独立把 wandb 接进 SFT 脚本了。
