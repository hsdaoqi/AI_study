"""
prepare_data.py —— 下载一个开源 SFT 数据集,转成 train_sft.py 吃的 messages 格式
================================================================================
背景:train_sft.py 吃的是每行一个 {"messages":[{"role","content"},...]} 的 jsonl。
     但大多数开源数据集是 Alpaca(instruction/input/output)等别的字段名,这里帮你转好。
     (对应 SFT任务_02 的 A1:各种格式本质都是"问答对",转成 messages 即可。)

用法(AutoDL,先挂镜像加速):
    source /etc/network_turbo                       # 学术加速(能直连 HF 最好)
    export HF_ENDPOINT=https://hf-mirror.com        # HF 国内镜像(强烈建议)
    pip install datasets                            # 若没装

    # 挑一个 preset 跑(得到 train.jsonl):
    python prepare_data.py --preset ruozhiba  --out train.jsonl                        # 弱智吧~1.5k(有趣,看效果爽)
    python prepare_data.py --preset alpaca_zh --out train.jsonl --max_samples 10000    # 中文指令(推荐从这起步)
    python prepare_data.py --preset belle     --out train.jsonl --max_samples 20000    # Belle 大规模取子集
    python prepare_data.py --preset dolly_en  --out train.jsonl                        # 英文人工高质量15k

    # 先小样本验证流程(强烈建议!确认能训起来再全量):
    python prepare_data.py --preset alpaca_zh --out tiny.jsonl --max_samples 200

然后拿它训:
    accelerate launch --multi_gpu --num_processes 2 train_sft.py \
        --data_path train.jsonl --epochs 3 --batch_size 4 --log_every 20
"""
import json
import argparse
from datasets import load_dataset


# —— 每个数据集字段名不同,写个小函数把一条样本抽成 (用户问, 助手答) ——
def alpaca_to_qa(ex):
    """Alpaca 式:instruction(+input) → 问;output → 答。Belle 也是这套。"""
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    user = instr if not inp else f"{instr}\n{inp}"
    return user, (ex.get("output") or "").strip()


def firefly_to_qa(ex):
    """Firefly:input → 问;target → 答。"""
    return (ex.get("input") or "").strip(), (ex.get("target") or "").strip()


def dolly_to_qa(ex):
    """Dolly:instruction(+context) → 问;response → 答。"""
    instr = (ex.get("instruction") or "").strip()
    ctx = (ex.get("context") or "").strip()
    user = instr if not ctx else f"{instr}\n{ctx}"
    return user, (ex.get("response") or "").strip()


# preset 名 → (HF 数据集名, split, 抽取函数)
PRESETS = {
    "ruozhiba":  ("LooksJuicy/ruozhiba",             "train", alpaca_to_qa),   # instruction/output
    "alpaca_zh": ("shibing624/alpaca-zh",            "train", alpaca_to_qa),   # instruction/input/output
    "belle":     ("BelleGroup/train_0.5M_CN",        "train", alpaca_to_qa),   # instruction/input/output
    "firefly":   ("YeungNLP/firefly-train-1.1M",     "train", firefly_to_qa),  # input/target
    "dolly_en":  ("databricks/databricks-dolly-15k", "train", dolly_to_qa),    # instruction/context/response
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS), help="选哪个数据集")
    ap.add_argument("--out", default="train.jsonl", help="输出 jsonl 路径")
    ap.add_argument("--max_samples", type=int, default=0, help="0=全部;否则只取前 N 条")
    ap.add_argument("--system", default="", help="给每条加一句 system 提示(可选)")
    args = ap.parse_args()

    name, split, to_qa = PRESETS[args.preset]
    print(f"下载数据集 {name} (split={split}) ...")
    ds = load_dataset(name, split=split)

    # 打印第一条的字段,方便你核对(万一数据集改了字段名,一眼看出来)
    print("原始样本字段示例:", {k: str(v)[:40] for k, v in ds[0].items()})

    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    print(f"共 {len(ds)} 条,开始转换成 messages 格式 ...")

    n_ok, n_skip = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            user, assistant = to_qa(ex)
            if not user or not assistant:          # 跳过空样本(也用来发现字段没对上)
                n_skip += 1
                continue
            msgs = []
            if args.system:
                msgs.append({"role": "system", "content": args.system})
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"✅ 写好 {args.out}:有效 {n_ok} 条,跳过空样本 {n_skip} 条")
    if n_ok == 0:
        print("⚠️ 全跳过了!多半是字段名对不上——看上面'原始样本字段示例',把对应的抽取函数改一下。")
    else:
        print(f"   下一步训练:accelerate launch --multi_gpu --num_processes 2 "
              f"train_sft.py --data_path {args.out} --epochs 3 --batch_size 4 --log_every 20")


if __name__ == "__main__":
    main()
