import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback


# =====================================================================
# 1. 复杂的自监督数据管道 (DataModule)
# =====================================================================
class SimCLRTransform:
    """模拟自监督的数据增强：对同一张图做两种不同的随机噪点处理，生成 View A 和 View B"""

    def __call__(self, x):
        # 模拟增强 1: 添加随机正态分布噪点
        view_a = x + torch.randn_like(x) * 0.1
        # 模拟增强 2: 添加随机均匀分布噪点并翻转
        view_b = torch.flip(x + (torch.rand_like(x) - 0.5) * 0.2, dims=[-1])
        return view_a, view_b


class ContrastiveDataset(Dataset):
    """模拟的自监督图像数据集"""

    def __init__(self, num_samples=10000):
        # 模拟 3通道, 64x64 的图像
        self.data = torch.randn(num_samples, 3, 64, 64)
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # 自监督训练返回：View A, View B
        view_a, view_b = self.transform(img)
        return view_a, view_b, idx  # 同时返回样本索引


class SimCLRDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # 模拟：只在主进程（Rank 0）下载或初始化大型原始文件
        # print("[Rank 0] 数据准备完毕。")
        pass

    def setup(self, stage: str = None):
        # 在所有 GPU 进程上，划分训练集和验证集
        entire_dataset = ContrastiveDataset(num_samples=20000)

        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = random_split(entire_dataset, [16000, 4000])
        if stage == "predict" or stage is None:
            # 预测阶段只需要单张原始图片，不需要增强视图
            self.predict_dataset = ContrastiveDataset(num_samples=2000)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def predict_dataloader(self):
        # 预测时单卡处理
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)


# =====================================================================
# 2. 复杂的对比学习模型 (LightningModule)
# =====================================================================
class SimCLRLightning(L.LightningModule):
    def __init__(self, lr: float = 1e-3, temperature: float = 0.07):
        super().__init__()
        self.save_hyperparameters()

        # 2.1 定义基础特征提取器 (ResNet-18)
        self.backbone = models.resnet18(weights=None)
        # 将 ResNet 的最后一层 FC 替换为投影头 (Projection Head, MLP)
        # 投影头在对比学习中能大幅提升表征质量
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # 将特征投影到 128 维低维空间进行对比
        )

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # x_i 和 x_j 分别是同一批图片的两个增强视图
        x_i, x_j, _ = batch

        # 2.2 前向传播提取低维投影特征
        z_i = self(x_i)  # 形状: (batch_size, 128)
        z_j = self(x_j)  # 形状: (batch_size, 128)

        # =========================================================
        # 核心：使用 self.all_gather 收集所有 GPU 的特征以计算全局 Contrastive Loss
        # =========================================================
        # 在 DDP 下，如果不 gather，每张卡只能和自己卡上的负样本对比。
        # 收集后，当前卡可以和所有卡上的负样本进行对比，对比学习的效果会成倍提升！
        all_z_i = self.all_gather(z_i)  # 形状: (num_gpus, batch_size, 128)
        all_z_j = self.all_gather(z_j)  # 形状: (num_gpus, batch_size, 128)

        # 将多维 Tensor 展平为二维
        all_z_i = all_z_i.view(-1, 128)  # 形状: (num_gpus * batch_size, 128)
        all_z_j = all_z_j.view(-1, 128)

        # 2.3 计算 InfoNCE 损失 (简化版对比损失)
        # 归一化特征
        all_z_i = F.normalize(all_z_i, dim=1)
        all_z_j = F.normalize(all_z_j, dim=1)

        # 计算所有样本之间的余弦相似度矩阵
        similarity_matrix = torch.matmul(all_z_i, all_z_j.T) / self.hparams.temperature

        # 对角线元素即为匹配的正样本对
        labels = torch.arange(similarity_matrix.size(0), device=self.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证集也计算对比损失
        x_i, x_j, _ = batch
        z_i = F.normalize(self(x_i), dim=1)
        z_j = F.normalize(self(x_j), dim=1)

        all_z_i = self.all_gather(z_i).view(-1, 128)
        all_z_j = self.all_gather(z_j).view(-1, 128)

        similarity_matrix = torch.matmul(all_z_i, all_z_j.T) / self.hparams.temperature
        labels = torch.arange(similarity_matrix.size(0), device=self.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    # =========================================================
    # 2.4 分布式推理预测 (Predict Loop)
    # =========================================================
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        在预测阶段，输入一个 batch 的原始图片，输出特征。
        Lightning 会在多卡上自动分配预测任务，并完美汇总。
        """
        view_a, _, idx = batch  # 此时只取第一个视图
        features = self(view_a)
        return {"idx": idx, "features": features}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)


# =====================================================================
# 3. 自定义高阶诊断回调函数 (Callback)
# =====================================================================
class AdvancedDiagnosticsCallback(Callback):
    """诊断回调：实时输出显存使用情况和每秒处理的样本数 (Throughput)"""

    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.time() - self.start_time
        local_rank = trainer.local_rank

        # 计算当前卡处理的样本数
        num_samples = len(trainer.train_dataloader.dataset) / trainer.world_size
        throughput = num_samples / duration

        # 获取显存使用
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 2)
            print(
                f"\n[GPU {local_rank}] Epoch {trainer.current_epoch} 完成 | 速度: {throughput:.1f} imgs/sec | 最大显存: {mem_used:.1f} MB")


# =====================================================================
# 4. 实验控制主程序
# =====================================================================
if __name__ == "__main__":
    # 初始化数据和模型
    dm = SimCLRDataModule(batch_size=128)
    model = SimCLRLightning(lr=1e-3)

    # 配置模型检查点（保存表现最好的3个模型）
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        dirpath="./simclr_checkpoints",
        filename="simclr-{epoch:02d}-{val_loss:.3f}"
    )

    # 4.1 训练阶段的 Trainer 配置
    trainer = L.Trainer(
        accelerator="gpu",
        devices=2,  # 【任务】：修改为你的实际 GPU 数量
        strategy="ddp",  # 采用分布式数据并行
        precision="16-mixed",  # 混合精度，降低显存并加速计算
        accumulate_grad_batches=2,  # 梯度累积：每 2 个 step 更新一次参数（虚拟扩大一倍 Batch Size）
        max_epochs=4,  # 训练 4 个 Epoch
        callbacks=[AdvancedDiagnosticsCallback(), checkpoint_callback],
        log_every_n_steps=10
    )

    print("=== 开始第一阶段：多卡分布式对比学习训练 ===")
    trainer.fit(model, datamodule=dm)

    print("\n=== 第二阶段：多卡分布式特征预测与提取 ===")
    # 重新加载刚才训练好的最好模型权重进行预测
    predictions = trainer.predict(model, datamodule=dm, ckpt_path="best")

    # 只在主进程（Rank 0）上打印预测特征结果
    if trainer.is_global_zero:
        print(f"预测流完成！共提取了 {len(predictions)} 个 batch 的特征。")
        # 打印第一个 batch 里的前 3 个特征向量样本
        print("样本特征维度样例:", predictions[0]["features"].shape)
        print("特征样例 (前3个样本):\n", predictions[0]["features"][:3])