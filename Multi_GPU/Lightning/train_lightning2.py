import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models  # 引入官方经典模型库
import lightning as L
import time


# =====================================================================
# 1. 定义一个重型模型 (使用标准的 ResNet-50)
# =====================================================================
class HeavyResNetClassifier(L.LightningModule):
    def __init__(self, num_classes: int = 10, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # 使用不带预训练权重的 ResNet-50，分类类别设为 10
        self.model = models.resnet50(num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# =====================================================================
# 2. 准备重型模拟数据 (3通道, 224x224 图像，类似 ImageNet 规格)
# =====================================================================
class LargeImageDataset(Dataset):
    def __init__(self, num_samples=20000):  # 2万张图片，足以跑一阵子了
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 动态生成单张图片数据，避免一次性加载过多数据导致 CPU 内存溢出
        x = torch.randn(3, 224, 224)
        y = torch.randint(0, 10, (1,)).item()
        return x, torch.tensor(y)


# =====================================================================
# 3. 启动逻辑
# =====================================================================
if __name__ == "__main__":
    train_dataset = LargeImageDataset(num_samples=30000)  # 训练集 3 万张
    val_dataset = LargeImageDataset(num_samples=5000)  # 验证集 5000 张

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # 每张卡分配 64 的 Batch Size。双卡就是 128
        shuffle=True,
        num_workers=4,  # 这里的物理 CPU 核心数，多卡时建议设为 4 或 8 避免数据加载成为瓶颈
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = HeavyResNetClassifier(lr=1e-3)

    # 4. 配置 Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=2,  # 我们先用 2 张卡跑，之后可以改成 1 张卡进行对比
        strategy="ddp",
        max_epochs=2,  # 跑 2 个 Epoch
        precision="16-mixed",  # 使用混合精度，降低显存占用并加速
        log_every_n_steps=10
    )

    start_time = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"--- 训练总耗时: {time.time() - start_time:.2f} 秒 ---")