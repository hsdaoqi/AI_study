# PyTorch 原生 DDP · 从零手写教程

---

## 目录
- [第一部分:先彻底看懂"单卡训练"](#第一部分先彻底看懂单卡训练)
- [第二部分:单卡→多卡,只改 6 个地方](#第二部分单卡多卡只改-6-个地方)
- [第三部分:6 处改动逐行手把手](#第三部分6-处改动逐行手把手)
- [第四部分:三个练习脚本(完整代码)](#第四部分三个练习脚本完整代码)
- [第五部分:实操顺序](#第五部分实操顺序)
- [第六部分:概念补充(心智模型/术语/all-reduce)](#第六部分概念补充)
- [第七部分:常见坑排查表](#第七部分常见坑排查表)
- [第八部分:自测题](#第八部分自测题)

---

# 第一部分:先彻底看懂"单卡训练"

**DDP 是在单卡训练的基础上改出来的。所以先把单卡的每一行看懂,这是地基。**

下面是一个最普通的单卡训练脚本(合成数据,能直接跑),我在每行后面标了它干嘛:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===== 1. 准备数据 =====
X = torch.randn(10000, 20)              # 10000 个样本,每个 20 维特征
y = torch.randint(0, 10, (10000,))      # 每个样本一个 0~9 的类别标签
ds = TensorDataset(X, y)                # 打包成 Dataset(把 X,y 配对)
loader = DataLoader(ds, batch_size=64, shuffle=True)  # 每次吐 64 个,打乱

# ===== 2. 准备模型 =====
model = nn.Sequential(                  # 一个最简单的 3 层 MLP
    nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 10)
).cuda()                                # .cuda() 把模型搬到 GPU 上

# ===== 3. 准备优化器和损失函数 =====
opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器:负责更新参数
loss_fn = nn.CrossEntropyLoss()                       # 分类任务的标准损失

# ===== 4. 训练循环 =====
for epoch in range(5):                  # 把整个数据集过 5 遍
    for xb, yb in loader:               # 每次拿一个 batch(64 个样本)
        xb, yb = xb.cuda(), yb.cuda()   # 数据也搬到 GPU
        opt.zero_grad()                 # 清空上一步的梯度
        out = model(xb)                 # 前向:算预测
        loss = loss_fn(out, yb)         # 算损失(预测 vs 真实标签)
        loss.backward()                 # 反向:算梯度
        opt.step()                      # 用梯度更新参数
    print(f'epoch {epoch} loss={loss.item():.4f}')
```

**这段你必须完全看懂**,因为 DDP 版本 90% 都和它一样。核心是训练循环里那 5 步:
`zero_grad → 前向 → 算loss → backward → step`,这是所有 PyTorch 训练的万能套路。

---

# 第二部分:单卡→多卡,只改 6 个地方

**好消息:上面那段单卡代码,要变成多卡 DDP,只需要加/改 6 个地方,其余一字不动。**

先给你一张"改动清单",看个全貌(下一部分逐行教怎么写):

| # | 改哪里 | 加什么 | 为了什么 |
|---|---|---|---|
| ① | 文件开头 | `dist.init_process_group('nccl')` | 启动多进程通信 |
| ② | 紧接着 | 读 rank/local_rank,`set_device` | 让每个进程认领自己的卡 |
| ③ | DataLoader | 加 `DistributedSampler` | 把数据切成 N 份,每卡吃不同的 |
| ④ | 模型后面 | `model = DDP(model, ...)` | 让梯度自动跨卡同步 |
| ⑤ | 循环里 | `sampler.set_epoch(epoch)` + `if rank==0` 打印 | 每轮重洗牌 + 只让0号打日志 |
| ⑥ | 结尾 | `if rank==0` 存模型 + `destroy_process_group()` | 只存一次 + 收尾 |

**记住这 6 个数字,你就记住了整个 DDP。** 下面一个一个手把手写。

---

# 第三部分:6 处改动逐行手把手

我们**从上面那段单卡代码出发**,一处一处地改。每处我都给你"**加在哪、写什么、为什么**"。

### 改动 ① + ②:文件最开头,初始化 + 绑卡

在所有代码**最前面**(import 之后),加这一段:

```python
import os                               # 新增:要读环境变量
import torch
import torch.nn as nn
import torch.distributed as dist        # 新增:分布式模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 新增
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler  # 加了 DistributedSampler

# ① 初始化进程组:建立各进程间的通信。'nccl' 是 GPU 通信后端。
dist.init_process_group(backend='nccl')

# ② 每个进程搞清楚"我是谁",并绑定到自己的卡
rank       = dist.get_rank()                     # 我的全局编号(0,1,2,...)
world_size = dist.get_world_size()               # 一共几个进程(=几张卡)
local_rank = int(os.environ['LOCAL_RANK'])       # 我在本机的卡号
torch.cuda.set_device(local_rank)                # 把我这个进程绑到 local_rank 号卡
```

**为什么要这样写?**
- `torchrun` 会开 N 个进程,each 从头跑一遍这个脚本。这几行让**每个进程知道自己是几号、该用哪张卡**。
- 不写 `set_device(local_rank)`,所有进程会挤在 0 号卡上,直接爆显存。

---

### 改动 ③:DataLoader 加 DistributedSampler

**原来的单卡写法:**
```python
loader = DataLoader(ds, batch_size=64, shuffle=True)
```

**改成:**
```python
# ③ DistributedSampler:把数据集切成 world_size 份,当前进程(rank)只拿其中一份
sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
loader = DataLoader(ds, batch_size=64, sampler=sampler)  # 注意:用了 sampler,就不能再写 shuffle=True
```

**为什么?**
- 2 张卡如果都吃一样的数据,那就是白算两遍,没意义。
- `DistributedSampler` 保证 **rank0 拿前一半、rank1 拿后一半(不重叠)**,这样才是真"并行"。
- ⚠️ 用了 sampler 就**必须去掉 `shuffle=True`**(打乱交给 sampler 做),否则报错。

---

### 改动 ④:模型用 DDP 包一层

**原来:**
```python
model = nn.Sequential(nn.Linear(20,128), nn.ReLU(), nn.Linear(128,10)).cuda()
```

**改成:**
```python
model = nn.Sequential(nn.Linear(20,128), nn.ReLU(), nn.Linear(128,10)).cuda()
# ④ 用 DDP 包住模型 —— 就这一行,让 backward 时梯度自动跨卡求平均
model = DDP(model, device_ids=[local_rank])
```

**为什么?**
- 这一行是 DDP 的**灵魂**。包了之后,你的训练循环**一个字都不用改**——`loss.backward()` 时,DDP 会自动在背后把各卡的梯度 all-reduce(求平均)。
- `device_ids=[local_rank]` 告诉 DDP 这个模型在哪张卡上。

---

### 改动 ⑤:训练循环里,set_epoch + 只让 rank0 打印

**原来:**
```python
for epoch in range(5):
    for xb, yb in loader:
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
    print(f'epoch {epoch} loss={loss.item():.4f}')
```

**改成(只加了 2 行,循环内部 5 步完全不变):**
```python
for epoch in range(5):
    sampler.set_epoch(epoch)            # ⑤-1 新增!每轮让 sampler 换种方式洗牌
    for xb, yb in loader:
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()                 # ← 梯度在这里被 DDP 自动同步(你无感)
        opt.step()
    if rank == 0:                       # ⑤-2 新增!只让 0 号进程打日志
        print(f'epoch {epoch} loss={loss.item():.4f}')
```

**为什么?**
- `sampler.set_epoch(epoch)`:不写的话,每个 epoch 数据切分/洗牌方式一模一样,等于没洗牌,影响训练效果。**这是新手最常忘的一行。**
- `if rank == 0`:不写的话,2 张卡会把日志打 2 遍,屏幕全是重复。

---

### 改动 ⑥:结尾,只让 rank0 存模型 + 收尾

在**最后**加:

```python
# ⑥ 只让 rank0 存模型;注意是 model.module(DDP 包了一层,真模型在 .module 里)
if rank == 0:
    torch.save(model.module.state_dict(), 'model.pth')
    print('saved model.pth (只有 rank0 保存)')

dist.destroy_process_group()            # 收尾,销毁进程组
```

**为什么?**
- `if rank == 0`:不写的话 N 张卡各存一个文件,互相覆盖还浪费。
- `model.module`:DDP 把你的模型包在了 `.module` 里。**存的时候要 `model.module.state_dict()`**,不然存出来的参数名会多一层 `module.` 前缀,以后加载对不上。

---

### ✅ 把 6 处拼起来 = 完整 DDP 脚本

上面 6 处改完,就是下面这个完整脚本(这就是 `代码/step03_train_ddp.py`):

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# ①② 初始化 + 绑卡
dist.init_process_group(backend='nccl')
rank       = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 数据
X = torch.randn(10000, 20)
y = torch.randint(0, 10, (10000,))
ds = TensorDataset(X, y)
# ③ DistributedSampler
sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
loader = DataLoader(ds, batch_size=64, sampler=sampler)

# 模型
model = nn.Sequential(nn.Linear(20,128), nn.ReLU(), nn.Linear(128,10)).cuda()
# ④ DDP 包一层
model = DDP(model, device_ids=[local_rank])

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(5):
    sampler.set_epoch(epoch)            # ⑤-1
    for xb, yb in loader:
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
    if rank == 0:                       # ⑤-2
        print(f'epoch {epoch} loss={loss.item():.4f}')

# ⑥ 存模型 + 收尾
if rank == 0:
    torch.save(model.module.state_dict(), 'model.pth')
    print('saved model.pth')
dist.destroy_process_group()
```

**运行:**
```bash
torchrun --nproc_per_node=2 step03_train_ddp.py
```

**看,DDP 版和单卡版对比,循环里那 5 步(zero_grad→前向→loss→backward→step)一模一样。** 你已经会写单卡,那 DDP 就只是"套一个固定的壳"。

---

# 第四部分:三个练习脚本(完整代码)

> 这三个脚本本目录 `代码/` 里也有,内容和下面完全一致。建议**从 step01 跑起**,一个个理解现象。

## 脚本 1:`step01_hello.py` —— 先证明"多进程真的起来了"

**目的**:最小代码,只看"torchrun 开了 N 个进程、每个有自己的 rank"。还没有模型、没有训练。

```python
import os
import torch
import torch.distributed as dist

def main():
    # ① 初始化进程组(读 torchrun 设好的环境变量)
    dist.init_process_group(backend='nccl')

    # ② 搞清楚"我是谁"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    # ③ 绑卡
    torch.cuda.set_device(local_rank)

    print(f'[rank {rank}/{world_size}] local_rank={local_rank} '
          f'-> GPU {torch.cuda.current_device()} '
          f'({torch.cuda.get_device_name(local_rank)})', flush=True)

    # ④ barrier:所有进程在此集合,演示同步
    dist.barrier()
    if rank == 0:
        print('==> 所有进程已汇合,DDP 世界启动成功。', flush=True)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```
**运行**:`torchrun --nproc_per_node=2 step01_hello.py`
**预期**:2 行 `[rank .../2]`,分别在 GPU0/GPU1;"已汇合"只打 1 次。

---

## 脚本 2:`step02_allreduce.py` —— 亲眼看到 all-reduce

**目的**:理解 DDP 同步梯度的底层操作。各卡各造一个值,all_reduce 求和,每张卡都拿到总和。

```python
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # 每张卡造一个只有自己知道的值:rank0->1.0, rank1->2.0
    t = torch.tensor([rank + 1.0], device='cuda')
    print(f'[rank {rank}] all_reduce 之前: {t.item()}', flush=True)

    # 核心:把所有卡上的 t 加起来,结果发回每张卡
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f'[rank {rank}] all_reduce(SUM) 之后: {t.item()}', flush=True)

    # 求平均 = 总和 / 卡数。DDP 对梯度就是这么做的。
    t /= world_size
    print(f'[rank {rank}] 求平均后: {t.item()}', flush=True)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```
**运行**:`torchrun --nproc_per_node=2 step02_allreduce.py`
**预期**:起始 1 和 2 → SUM 后两卡都变 3 → 平均后都变 1.5。

---

## 脚本 3:`step03_train_ddp.py` —— 完整训练

即第三部分最后那段完整代码。带注释的加强版:

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        return self.net(x)


def main():
    # ①② 初始化 + 绑卡
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    is_master = (rank == 0)

    # 数据(所有进程用同一 seed 造同一份,靠 sampler 切分)
    torch.manual_seed(0)
    X = torch.randn(10000, 20)
    y = torch.randint(0, 10, (10000,))
    ds = TensorDataset(X, y)

    # ③ 切数据
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)

    # ④ DDP 包模型
    model = ToyNet().cuda()
    model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)        # ⑤-1
        model.train()
        n_batches, total = 0, 0.0
        for xb, yb in loader:
            xb = xb.cuda(non_blocking=True)
            yb = yb.cuda(non_blocking=True)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()             # 梯度在此自动 all-reduce
            opt.step()
            total += loss.item()
            n_batches += 1
        if is_master:                   # ⑤-2
            print(f'[epoch {epoch}] rank0 处理 {n_batches} 个 batch, '
                  f'平均 loss={total/n_batches:.4f}', flush=True)

    # ⑥ 存模型 + 收尾
    if is_master:
        torch.save(model.module.state_dict(), 'toynet_ddp.pth')
        print('==> 已保存 toynet_ddp.pth(只有 rank0 保存)', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
```
**运行**:
```bash
torchrun --nproc_per_node=1 step03_train_ddp.py   # 单卡
torchrun --nproc_per_node=2 step03_train_ddp.py   # 双卡对比
```
**预期**:双卡时每 epoch 的 batch 数约为单卡一半(数据被切两半并行);只有 rank0 存模型。

---

# 第五部分:实操顺序

> 前置:租 AutoDL **2 卡**实例;把 `代码/` 传上去;`nvidia-smi` 看到 2 块卡;`python -c "import torch;print(torch.cuda.device_count())"` 打印 2。

```bash
cd /root/autodl-tmp/代码           # 或你放脚本的目录

# 第 1 步:证明多进程起来了
torchrun --nproc_per_node=2 step01_hello.py

# 第 2 步:看懂 all-reduce
torchrun --nproc_per_node=2 step02_allreduce.py

# 第 3 步:完整训练(先单卡再双卡)
torchrun --nproc_per_node=1 step03_train_ddp.py
torchrun --nproc_per_node=2 step03_train_ddp.py
```

**每跑一个,先对照上面的"预期"看现象对不对,再进下一个。有报错查第七部分。**

---

# 第六部分:概念补充

## 心智模型(一句话)
每张卡放一份完整模型,吃不同数据,反向传播时把梯度求平均,保证 N 份模型始终一致。
= "用多卡模拟一个 N 倍大的 batch",快 N 倍,结果等价。

## 6 个术语
| 术语 | 含义 | 记忆点 |
|---|---|---|
| process | DDP 开 N 个进程,每进程独占一卡 | 多进程不是多线程 |
| rank | 全局编号 0~N-1 | rank0 = master,打日志/存模型 |
| local_rank | 本机卡号 | 用于 `set_device` 绑卡 |
| world_size | 总进程数=总卡数 | 梯度平均的分母 |
| process group | 进程通信集体 | init_process_group 建立 |
| backend | 通信后端 | GPU 用 nccl |

## all-reduce(唯一的魔法)
把所有卡的某张量求和/求平均,再发回每张卡。DDP 在 `loss.backward()` 时自动对梯度做这个。你在 VAR `quant.py` 见过的 `tdist.all_reduce(hit_V)` 就是它。

## torchrun 做了什么
开 N 个进程、each 跑一遍脚本、设好 `RANK/WORLD_SIZE/LOCAL_RANK` 环境变量。单机多卡你基本只改 `--nproc_per_node`。

---

# 第七部分:常见坑排查表

| 现象 | 原因 | 解决 |
|---|---|---|
| `RANK not set` 报错 | 直接 `python xxx.py` 跑了 | 必须用 `torchrun` 启动 |
| 卡住 / NCCL timeout | 端口被占 | 加 `--master_port=29501` |
| `Address already in use` | 上次进程没退干净 | 换端口 或 `pkill -f torchrun` |
| 所有进程挤 GPU0、爆显存 | 忘了 `set_device(local_rank)` | 加上绑卡那行 |
| `DataLoader` 报 sampler/shuffle 冲突 | 同时写了 sampler 和 shuffle=True | 用 sampler 就删 shuffle=True |
| 存的模型 key 多 `module.` | 存了 DDP 包装层 | 存 `model.module.state_dict()` |
| loss 不收敛/每轮一样 | 忘了 `sampler.set_epoch(epoch)` | 每 epoch 开头调一次 |
| 日志打 N 遍/存 N 个文件 | 没做 rank0 判断 | 用 `if rank == 0:` |
| 多卡比单卡还慢 | 模型太小,通信开销>计算 | 正常,小模型不划算,大模型才见效 |

---

# 第八部分:自测题

答得出说明真懂了(答案在教程里都有):

1. 把单卡训练改成 DDP,一共要改哪 6 个地方?(不看清单默写)
2. 训练循环里的 5 步(zero_grad→前向→loss→backward→step),DDP 版和单卡版有区别吗?梯度同步发生在哪一步?
3. `world_size`、`rank`、`local_rank` 单机 4 卡时各是多少?
4. 为什么存模型要 `model.module.state_dict()`?
5. 不写 `sampler.set_epoch(epoch)` 会怎样?
6. 为什么必须用 `torchrun` 而不能 `python xxx.py`?
7. DataLoader 用了 DistributedSampler 后,为什么不能再写 `shuffle=True`?
