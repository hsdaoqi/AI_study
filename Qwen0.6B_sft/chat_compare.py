"""
chat_compare.py —— 对比微调前后的效果(原始 Qwen3-0.6B  vs  你 SFT 出来的模型)
============================================================================
它把两个模型都加载进来,对同一批问题各答一遍,并排打印,让你一眼看出差别。

用法(单卡就行,不用 accelerate):
    python chat_compare.py
    # 默认:base=Qwen/Qwen3-0.6B(微调前), tuned=output/qwen3-sft(微调后)

    python chat_compare.py --interactive          # 跑完固定问题后,进入对话模式自己问
    python chat_compare.py --tuned ""             # 只看微调前(原始模型)
    python chat_compare.py --base ""              # 只看微调后
    python chat_compare.py --max_new_tokens 256   # 让它多答点

注意:tuned 默认是 output/qwen3-sft(相对路径)。你在哪个目录跑训练、模型就存在哪,
     所以在同一个目录(week2/)跑这个脚本就对得上。
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 一组测试问题(覆盖:自我介绍/翻译/常识/数学/写作)。想加自己的,往这个列表里塞。
TEST_QUESTIONS = [
    "用一句话介绍你自己。",
    "把'今天天气很好'翻译成英文。",
    "水在标准大气压下的沸点是多少?",
    "3 乘以 4 等于多少?",
    "推荐一种放松方式。",
    "写一句鼓励人努力的话。",
    "用简单的话解释什么是人工智能。",
    "北京是哪个国家的首都?",
]


def load(model_path):
    """加载一个模型 + 它的分词器,放到 GPU、切到推理模式。"""
    print(f"加载模型:{model_path} ...")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


@torch.no_grad()                       # 推理不需要梯度,省显存
def answer(tok, model, question, max_new_tokens):
    """把一个问题喂给模型,返回它生成的回答(只取新生成的部分)。"""
    msgs = [{"role": "user", "content": question}]
    # 用 tokenize=False 拿字符串再编码,避开新版 apply_chat_template 返回 Encoding 的坑;
    # enable_thinking=False:关掉 Qwen3 的"思考模式",直接给答案,对比更干净。
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:                  # 老版本 template 不认 enable_thinking,就不传
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,               # 贪心解码:结果可复现,对比公平
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    # out 包含了"问题+回答",切掉前面的问题,只留模型新生成的回答
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B", help="微调前的原始模型;设为空串跳过")
    ap.add_argument("--tuned", default="output/qwen3-sft", help="微调后的模型目录;设为空串跳过")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--interactive", action="store_true", help="跑完固定问题后进入对话模式")
    args = ap.parse_args()

    base = load(args.base) if args.base else None
    tuned = load(args.tuned) if args.tuned else None

    # —— 固定问题,并排对比 ——
    for q in TEST_QUESTIONS:
        print("\n" + "=" * 72)
        print(f"❓ {q}")
        if base:
            print(f"\n[微调前] {answer(*base, q, args.max_new_tokens)}")
        if tuned:
            print(f"\n[微调后] {answer(*tuned, q, args.max_new_tokens)}")
    print("\n" + "=" * 72)

    # —— 对话模式:自己输入问题,回车提交,空行退出 ——
    if args.interactive:
        print("\n进入对话模式(直接输入问题,空行退出):")
        while True:
            try:
                q = input("\n你:").strip()
            except EOFError:
                break
            if not q:
                break
            if base:
                print(f"[微调前] {answer(*base, q, args.max_new_tokens)}")
            if tuned:
                print(f"[微调后] {answer(*tuned, q, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
