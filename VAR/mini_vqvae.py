"""
mini_vqvae.py  —— 迷你 VQVAE:把"蜡笔理论"变成能跑、能看的代码
================================================================
这个脚本干嘛:在 MNIST(手写数字)上训练一个最小的 VQVAE,
每个 epoch 存一张对比图(上排=原图,下排=重建图),你能亲眼看着"重建"从糊变清晰。

它把你学过的两块知识合体了:
  · VQVAE 理论:码本 / 挑蜡笔(量化)/ 三个练习(3个loss)/ STE     → 见下面 VectorQuantizer 和训练循环
  · DDP 多卡:你学的那 6 处改动,这次套在真模型上                  → 见标了 ①~⑥ 的地方

怎么跑:
  单卡: python mini_vqvae.py
  双卡: torchrun --nproc_per_node=2 mini_vqvae.py
  (同一个脚本,自动判断。下载慢就先 `source /etc/network_turbo`)

依赖:torch、torchvision。若缺:pip install torchvision
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ============================================================
# 一、模型:VQVAE = 编码器 + 码本(蜡笔盒) + 解码器
# ============================================================

class Encoder(nn.Module):
    """① 眯眼睛:把 28x28 的图,概括成 7x7 的小网格,每格 D 个数字(z_e)"""
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),   # 28 -> 14
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 14 -> 7
            nn.Conv2d(64, D, 3, stride=1, padding=1),              # 7x7, 每格 D 维 = z_e
        )
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """③ 照着挑好的蜡笔,把 7x7 小网格放大还原成 28x28 图"""
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(D, 64, 3, stride=1, padding=1), nn.ReLU(),  # 7x7
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(), # 7 -> 14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),             # 14 -> 28
            nn.Sigmoid(),   # 输出压到 [0,1],和 MNIST 像素对齐
        )
    def forward(self, z_q):
        return self.net(z_q)


class VectorQuantizer(nn.Module):
    """② 挑蜡笔(量化)—— 这里就是全篇的心脏:码本 / argmin / 两个loss / STE"""
    def __init__(self, K, D, beta=0.25):
        super().__init__()
        self.K, self.D, self.beta = K, D, beta
        # 码本 = 一盒 K 支蜡笔,每支 D 维。这就是"蜡笔盒 e"
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)

    def forward(self, z_e):                 # z_e: (B, D, H, W) = 你"想要的颜色"
        B, D, H, W = z_e.shape
        # 把每个格子的向量摊平成一列:(B*H*W, D),一格一个"想要的颜色"
        z = z_e.permute(0, 2, 3, 1).reshape(-1, D)

        # —— 挑最接近的蜡笔(argmin)——
        # 算每个"想要的颜色"到 K 支蜡笔的距离,取最近那支的编号
        dist = (z**2).sum(1, keepdim=True) \
             - 2 * z @ self.embedding.weight.t() \
             + (self.embedding.weight**2).sum(1)
        idx = dist.argmin(1)                              # 每格挑中的蜡笔编号(token!)
        z_q = self.embedding(idx).view(B, H, W, D).permute(0, 3, 1, 2)  # 换成蜡笔 -> z_q

        # —— 两个练习(两个loss)——
        # 练习2 codebook loss:冻住 z_e,只挪蜡笔 -> "蜡笔追你"
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        # 练习3 commitment loss:冻住蜡笔,只挪 z_e -> "你追蜡笔",别乱飘
        commit_loss   = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # —— STE 直通:正向用挑好的蜡笔 z_q,反向让梯度假装没挑、直穿回 encoder ——
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss, idx


class MiniVQVAE(nn.Module):
    def __init__(self, K=128, D=64):
        super().__init__()
        self.enc = Encoder(D)
        self.vq  = VectorQuantizer(K, D)
        self.dec = Decoder(D)
    def forward(self, x):
        z_e = self.enc(x)               # ① 眯眼睛
        z_q, vq_loss, idx = self.vq(z_e)  # ② 挑蜡笔
        recon = self.dec(z_q)           # ③ 画回去
        return recon, vq_loss, idx


# ============================================================
# 二、训练:和你跑过的 step03 一模一样的 5 步骨架,只换了模型和loss
# ============================================================

def main():
    EPOCHS = 10
    BATCH  = 128

    # ①② DDP:自动判断是不是 torchrun 启动的。是 -> 多卡;不是 -> 单卡
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend='nccl')              # ① 初始化进程组
        rank       = dist.get_rank()                         # ② 我是谁
        world      = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)                    #    绑到自己的卡
        device = f'cuda:{local_rank}'
        is_master = (rank == 0)
    else:
        rank, world, local_rank = 0, 1, 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        is_master = True

    # 数据:MNIST,自动下载。ToTensor 把像素变成 [0,1]
    tf = transforms.ToTensor()
    ds = datasets.MNIST(root='./data', train=True, download=True, transform=tf)

    # ③ DistributedSampler:多卡时把数据切成 world 份,每卡吃不同的一份
    if ddp:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
        loader  = DataLoader(ds, batch_size=BATCH, sampler=sampler, num_workers=2)
    else:
        loader  = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2)

    # ④ 模型;多卡时用 DDP 包一层(梯度自动跨卡同步)
    model = MiniVQVAE(K=128, D=64).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    core = model.module if ddp else model   # 真身(存模型/取组件时用它)

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(EPOCHS):
        if ddp:
            sampler.set_epoch(epoch)         # ⑤-1 每轮重新洗牌(别忘!)
        model.train()
        total_recon, total_vq, n = 0.0, 0.0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            # ===== 万能 5 步骨架(和 step03 一样)=====
            opt.zero_grad()                          # 清零
            recon, vq_loss, _ = model(imgs)          # 前向
            recon_loss = F.mse_loss(recon, imgs)     #   练习1:重建 loss
            loss = recon_loss + vq_loss              #   总 loss = 练习1 + (练习2+练习3)
            loss.backward()                          # 反向(STE 让梯度穿过量化)
            opt.step()                               # 更新
            # =======================================
            total_recon += recon_loss.item(); total_vq += vq_loss.item(); n += 1

        if is_master:                                # ⑤-2 只让 0 号卡打日志
            print(f'[epoch {epoch}] recon_loss={total_recon/n:.4f}  vq_loss={total_vq/n:.4f}', flush=True)
            # 存一张对比图:上排原图,下排重建图 —— 你能亲眼看它一轮轮变清晰
            model.eval()
            with torch.no_grad():
                recon, _, _ = core(imgs[:8])
            grid = torch.cat([imgs[:8].cpu(), recon[:8].cpu()], dim=0)  # 16 张:8原 + 8重建
            save_image(grid, f'recon_epoch{epoch}.png', nrow=8)

    # ⑥ 收尾:只让 0 号卡存模型;销毁进程组
    if is_master:
        torch.save(core.state_dict(), 'mini_vqvae.pth')
        print('==> 训练完成。看 recon_epoch*.png 对比图,mini_vqvae.pth 是权重。', flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
