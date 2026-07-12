"""
train_sft.py —— Qwen3-0.6B 监督微调(SFT)· 双卡 Accelerate + wandb 记录
======================================================================
这是师兄 SFT 任务的第③项:训练脚本本体。它把你前面三份笔记全用上了:
  · SFT任务_01_HF模型文件格式.md  → 第①块:加载模型/分词器(from_pretrained)
  · SFT任务_02_数据集格式与Loss.md → 第②块:messages→ChatML→input_ids/labels(-100 掩码)
                                      → 第④块:out.loss = HF 内部 shift + 交叉熵(就是笔记 Part B)
  · SFT任务_03_wandb详解.md        → 第③块:Accelerate 原生 tracker(Part F3)+ 离线开关(Part D2)
                                      → wandb.watch 权重/梯度直方图(Part C5)+ tag(Part D1)

它能干嘛:
  · 双卡(或任意多卡)数据并行,框架用 HuggingFace Accelerate(不是裸 DDP,不是 Megatron)
  · 内置一份迷你 demo 数据(单轮对话),不用下数据集,离线也能直接跑通
  · 也支持 --data_path 读你自己的 jsonl(每行一个 {"messages":[...]})
  · loss 只在 assistant 回答上算(其余 -100),和笔记 A4 完全一致
  · wandb 实时曲线(train/loss、train/lr)+ 权重/梯度直方图;在线/离线一键切换,自动打 tag

怎么跑(AutoDL,双卡)——见文件最底部【运行说明】
"""

import os
import json
import math
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed


# =====================================================================
# 0. 内置 demo 数据(单轮对话,messages 格式 —— 见 SFT任务_02 的 A1③)
#    目的:不下任何数据集就能把脚本跑通。想看"学没学到",我故意让所有回答
#    都带一个固定签名"—— 道柒的小助手",训几轮后模型应该学会在结尾带上它。
#    换成你自己的数据:用 --data_path xxx.jsonl,每行一个 {"messages":[...]}。
# =====================================================================
DEMO_DATA = [
    {"messages": [
        {"role": "system", "content": "你是一个简洁、友好的中文助手。"},
        {"role": "user", "content": "1+1等于几?"},
        {"role": "assistant", "content": "等于 2。—— 道柒的小助手"},
    ]},
    {"messages": [
        {"role": "user", "content": "用一句话介绍你自己。"},
        {"role": "assistant", "content": "我是一个乐于助人的中文助手。—— 道柒的小助手"},
    ]},
    {"messages": [
        {"role": "user", "content": "水的沸点是多少?"},
        {"role": "assistant", "content": "标准大气压下是 100 摄氏度。—— 道柒的小助手"},
    ]},
    {"messages": [
        {"role": "user", "content": "把'今天天气很好'翻译成英文。"},
        {"role": "assistant", "content": "The weather is nice today. —— 道柒的小助手"},
    ]},
    {"messages": [
        {"role": "user", "content": "推荐一种放松方式。"},
        {"role": "assistant", "content": "出去散散步,晒晒太阳。—— 道柒的小助手"},
    ]},
    {"messages": [
        {"role": "system", "content": "你是一个简洁、友好的中文助手。"},
        {"role": "user", "content": "Python 里怎么打印一句话?"},
        {"role": "assistant", "content": "用 print(\"你好\") 就行。—— 道柒的小助手"},
    ]},
    {"messages": [
        {"role": "user", "content": "3 乘以 4 等于多少?"},
        {"role": "assistant", "content": "等于 12。—— 道柒的小助手"},
    ]},
    {"messages": [
        {"role": "user", "content": "早上好!"},
        {"role": "assistant", "content": "早上好呀,今天也要加油!—— 道柒的小助手"},
    ]},
]


def load_data(path):
    """有 --data_path 就读 jsonl(每行一个 {"messages":[...]}),否则用内置 demo。"""
    if path and os.path.exists(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return DEMO_DATA


# =====================================================================
# 1. Dataset:一条 messages → input_ids + labels(loss 掩码在这里做)
#    完全对应 SFT任务_02 的 A2/A4:
#      · apply_chat_template 把 messages 拼成 ChatML 字符串再切成 token
#      · 只有"assistant 回答 + 结尾 <|im_end|>"保留真实 id,其余全填 -100
#    技巧:prompt = 到"该助手答了"为止(add_generation_prompt=True),它的长度
#          就是要屏蔽的前缀长度;full = 含答案的完整序列。
#          labels 前 len(prompt) 个位置设 -100,剩下的(答案+im_end)保留。
# =====================================================================
class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        msgs = self.samples[idx]["messages"]

        # 完整对话(含最后一句 assistant 回答),切成 token id 列表
        full_ids = self.tok.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        # 只到"<|im_start|>assistant\n"为止的前缀(不含答案内容)
        prompt_ids = self.tok.apply_chat_template(
            msgs[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

        labels = list(full_ids)
        # 前缀(system+user+assistant的头)全部 -100 → 不算 loss(见 A4)
        n_mask = min(len(prompt_ids), len(labels))
        for i in range(n_mask):
            labels[i] = -100

        # 截断到 max_len(demo 很短,用不上;长数据防炸显存)
        full_ids = full_ids[: self.max_len]
        labels = labels[: self.max_len]

        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    """
    组 batch:右侧 padding 对齐(见 SFT任务_02 的 A6)
      · input_ids   短的用 pad_id 补齐
      · attention_mask 真 token=1,padding=0(让注意力忽略 padding)
      · labels      padding 位置也填 -100(padding 不算 loss)
    """
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attn, labels = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attn.append([1] * n + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# =====================================================================
# 2. 命令行参数(全给默认值,直接 accelerate launch train_sft.py 就能跑)
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    # —— 模型/数据 ——
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF 模型 id 或本地路径")
    p.add_argument("--data_path", default="", help="jsonl 数据路径;留空用内置 demo")
    p.add_argument("--max_len", type=int, default=1024, help="单条最大 token 数")
    p.add_argument("--out_dir", default="output/qwen3-sft", help="模型保存目录")
    # —— 训练超参 ——
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2, help="每张卡的 batch(demo小,设2)")
    p.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪;<=0 关闭")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=1, help="每多少个更新步 log 一次")
    # —— wandb(见 SFT任务_03)——
    p.add_argument("--project", default="qwen3-sft", help="wandb 项目名")
    p.add_argument("--run_name", default="qwen3-0.6b-sft", help="这次 run 的名字")
    p.add_argument("--tags", default="sft,qwen3-0.6b,2gpu,accelerate",
                   help="逗号分隔的 tag(见 Part D1)")
    p.add_argument("--wandb_mode", default="", choices=["", "online", "offline", "disabled"],
                   help="在线/离线/关闭;留空则听环境变量 WANDB_MODE(见 Part D2)")
    p.add_argument("--wandb_dir", default="", help="wandb 本地文件目录(离线时建议放数据盘)")
    p.add_argument("--no_wandb", action="store_true", help="彻底不接 wandb(纯本地调试)")
    p.add_argument("--watch_model", action="store_true",
                   help="开 wandb.watch 记录权重/梯度直方图(见 Part C5;略耗时)")
    return p.parse_args()


# =====================================================================
# 3. 主流程
# =====================================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    # —— wandb 在线/离线开关:在建 Accelerator 之前设环境变量最稳(见 Part D2/E)——
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode        # online/offline/disabled
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir          # 离线文件放哪(别塞满系统盘)

    # —— Accelerator:log_with="wandb" 声明用 wandb 当 tracker(见 Part F3)——
    #    不传 mixed_precision:我们直接把模型加载成 bf16(纯 bf16 训练),
    #    省显存、和 inspect_qwen.py 里学的 torch_dtype=bfloat16 一致。
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        log_with=None if args.no_wandb else "wandb",
    )

    def rprint(*a, **k):
        """只在主进程打印(多卡时别刷 N 份,和你 DDP 学的 if rank==0 一个道理)。"""
        accelerator.print(*a, **k)

    rprint("=" * 60)
    rprint(f"进程数(卡数) = {accelerator.num_processes} | 设备 = {accelerator.device}")
    rprint(f"模型 = {args.model} | epochs={args.epochs} bs/卡={args.batch_size} lr={args.lr}")
    rprint("=" * 60)

    # —— ① 加载分词器 + 模型(对应 SFT任务_01 / inspect_qwen.py 第5步)——
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # Qwen 默认有 pad;保险起见兜底
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False   # 训练时关掉 KV cache(推理才用,训练开着白占显存/报警告)

    # —— ② 数据 → Dataset → DataLoader(对应 SFT任务_02 A2/A4/A6)——
    samples = load_data(args.data_path)
    rprint(f"数据条数 = {len(samples)}(来源:{'内置 demo' if not args.data_path else args.data_path})")
    dataset = SFTDataset(samples, tokenizer, args.max_len)

    from functools import partial
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_id=tokenizer.pad_token_id),
    )

    # —— 优化器(AdamW),学习率调度器等 prepare 之后按真实步数建 ——
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # —— init_trackers:自动只在主进程建 wandb run(见 Part F3)——
    #    config 存超参(Part C1/C4);tags 打标签(Part D1②)。
    if not args.no_wandb:
        accelerator.init_trackers(
            project_name=args.project,
            config={
                "model": args.model,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "epochs": args.epochs,
                "max_len": args.max_len,
                "num_gpus": accelerator.num_processes,
            },
            init_kwargs={"wandb": {
                "name": args.run_name,
                "tags": [t for t in args.tags.split(",") if t],
            }},
        )

    # —— prepare:把 model/optimizer/loader 交给 Accelerate 包装(自动分数据到各卡)——
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # prepare 之后 loader 已按卡切分,用它的长度算真实总步数,再建调度器
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.03 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    rprint(f"每 epoch 更新步 = {steps_per_epoch} | 总更新步 = {total_steps} | warmup = {warmup_steps}")

    # —— wandb.watch:权重/梯度直方图(见 Part C5;对应任务里'权重曲线'的另一种理解)——
    #    只主进程开;要传"没被 DDP 包过"的原始模型,用 unwrap_model。
    if args.watch_model and not args.no_wandb and accelerator.is_main_process:
        try:
            import wandb
            wandb.watch(accelerator.unwrap_model(model), log="all", log_freq=args.log_every)
            rprint("已开启 wandb.watch(权重/梯度直方图)")
        except Exception as e:
            rprint("wandb.watch 跳过:", e)

    # =================================================================
    # 4. 训练循环 —— 和你 DDP step03 / mini_vqvae 里那个 5 步循环一模一样:
    #    forward → loss → backward → step → zero_grad,只是模型换成了 Qwen。
    # =================================================================
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                # forward:传 labels,HF 内部自动 shift + 交叉熵(ignore_index=-100)
                # 这一行 out.loss 就是 SFT任务_02 Part B 推的那个 -log 平均值
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss

                accelerator.backward(loss)     # 反向;多卡时自动 all-reduce 同步梯度

                # 梯度裁剪:只在"真正要更新"的那一步做(累积够了)
                if accelerator.sync_gradients and args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()               # 更新(Accelerate 会自动只在累积边界生效)
                optimizer.zero_grad()
                if accelerator.sync_gradients: # 调度器只在真正更新时走一步
                    scheduler.step()

            # sync_gradients=True 表示这一步真的更新了参数 → 才算一个 global_step
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    # 把各卡的 loss 收集起来求平均,得到一个干净的数(见 Part H)
                    avg_loss = accelerator.gather(loss.detach()).mean().item()
                    cur_lr = scheduler.get_last_lr()[0]
                    # accelerator.log 自动只在主进程记;键名用 / 前缀分组(见 Part D1①)
                    if not args.no_wandb:
                        accelerator.log(
                            {"train/loss": avg_loss,
                             "train/lr": cur_lr,
                             "train/epoch": epoch},
                            step=global_step,
                        )
                    rprint(f"epoch {epoch} | step {global_step}/{total_steps} "
                           f"| loss {avg_loss:.4f} | lr {cur_lr:.2e}")

    # =================================================================
    # 5. 保存模型(只主进程)+ 收尾
    # =================================================================
    accelerator.wait_for_everyone()           # 等所有卡都训完
    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)     # 脱掉 DDP 外壳再存
        unwrapped.save_pretrained(args.out_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(args.out_dir)         # 分词器也一起存(推理要用)
        rprint(f"✅ 模型已保存到 {args.out_dir}")

    if not args.no_wandb:
        accelerator.end_training()            # = wandb.finish(),把缓冲刷完正常关 run

    rprint("训练结束。")


if __name__ == "__main__":
    main()


# =====================================================================
# 【运行说明】AutoDL 双卡
# ---------------------------------------------------------------------
# 0) 装依赖(第一次):
#    pip install "transformers>=4.51" accelerate wandb safetensors
#
# 1) 先下模型(可选,提前缓存;也可直接训练时自动下):
#    source /etc/network_turbo                      # 学术加速
#    export HF_ENDPOINT=https://hf-mirror.com       # 若还慢,用镜像
#    python inspect_qwen.py                          # 顺便验一遍模型文件
#
# 2-A) 在线跑(能连外网,wandb 实时上传 —— 师兄说的"优先动态上传"):
#    wandb login                                     # 首次粘 API key(wandb.ai/authorize)
#    accelerate launch --multi_gpu --num_processes 2 train_sft.py --watch_model
#
# 2-B) 离线跑(AutoDL 连不上 wandb 云 —— 退路,见 Part D2):
#    export WANDB_DIR=/root/autodl-tmp/wandb         # 离线文件放数据盘,别塞满系统盘
#    accelerate launch --multi_gpu --num_processes 2 train_sft.py \
#        --wandb_mode offline --wandb_dir /root/autodl-tmp/wandb --watch_model
#    # 训完,有网时把离线记录补传上去:
#    wandb login
#    wandb sync /root/autodl-tmp/wandb/offline-run-*
#
# 3) 用自己的数据(每行一个 {"messages":[...]} 的 jsonl):
#    accelerate launch --multi_gpu --num_processes 2 train_sft.py \
#        --data_path /root/autodl-tmp/my_sft.jsonl --epochs 3 --batch_size 4
#
# 4) 单卡快速调试(先确认脚本不报错,再上双卡):
#    python train_sft.py --no_wandb --epochs 1
#
# 小抄:
#   --no_wandb     完全不接 wandb(纯本地 debug)
#   --wandb_mode   online / offline / disabled
#   --watch_model  记录权重/梯度直方图(不加则只记 loss/lr 曲线)
#   --num_processes N   用几张卡(双卡就是 2)
# =====================================================================
